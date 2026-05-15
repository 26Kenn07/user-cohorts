import ast
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from db.milvus import upsert_video_embeddings, upsert_user_embeddings
from utils.engagement import compute_engagement_scores
from utils.embeddings import embed_videos
from utils.cohort import cluster_videos, assign_user_cohorts, build_cohort_profiles
from utils.prompt_generator import generate_all_labels
from models.two_tower import (
    TwoTowerModel,
    EngagementDataset,
    IndexMaps,
    train,
    get_video_embeddings_finetuned,
    get_user_embedding,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

CSV_FILE         = "new_ck_user_events.csv"
VIDEO_CSV_FILE   = "new_video_data.csv"
MIN_INTERACTIONS = 10
TEST_RATIO       = 0.2


def _save(name: str, obj: object) -> None:
    with open(CACHE_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Cached {name}")


def _load(name: str) -> object | None:
    path = CACHE_DIR / f"{name}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Loaded {name} from cache")
        return obj
    return None


def load_events() -> pd.DataFrame:
    cached = _load("all_events")
    if cached is not None:
        return cached  # type: ignore[return-value]
    logger.info(f"Loading events from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    # Strip stray quotes that ClickHouse sometimes wraps around IDs
    df["user_id"] = df["user_id"].astype(str).str.strip('"')
    df["video_id"] = df["video_id"].astype(str).str.strip('"')
    logger.info(f"  {len(df)} rows, {df['user_id'].nunique()} unique users")
    _save("all_events", df)
    return df


def get_videos(video_ids: list[str]) -> list[dict]:
    cached = _load("videos")
    if cached is not None:
        return cached  # type: ignore[return-value]
    logger.info(f"Loading videos from {VIDEO_CSV_FILE}...")
    df = pd.read_csv(VIDEO_CSV_FILE)
    df = df[df["video_id"].isin(set(video_ids))]

    def _parse_keywords(val: object) -> list[str]:
        if pd.isna(val) or not val:  # type: ignore[arg-type]
            return []
        try:
            parsed = ast.literal_eval(str(val))
            if isinstance(parsed, list):
                return [str(k) for k in parsed]
        except (ValueError, SyntaxError):
            pass
        return [s.strip() for s in str(val).split(",") if s.strip()]

    def _str(val) -> str:
        if pd.isna(val) if not isinstance(val, (list, dict)) else False:
            return ""
        s = str(val).strip()
        return "" if s.lower() == "nan" else s

    videos: list[dict] = []
    for _, row in df.iterrows():
        videos.append({
            "video_id": str(row["video_id"]),
            "transcript": _str(row.get("transcript")),
            "description_text": _str(row.get("description_text")),
            "video_gen_description": _str(row.get("video_gen_description")),
            "keywords": _parse_keywords(row.get("keywords")),
            "embedding_confidence": _str(row.get("embedding_confidence")) or "none",
        })

    logger.info(f"Loaded {len(videos)} videos from CSV (of {len(video_ids)} requested)")
    _save("videos", videos)
    return videos


def get_video_embeddings_cached(videos: list[dict]) -> dict:
    cached = _load("video_embeddings")
    if cached is not None:
        return cached  # type: ignore[return-value]
    embs = embed_videos(videos)
    _save("video_embeddings", embs)
    return embs


def per_user_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-user chronological split: oldest 80% → train, newest 20% → test.
    Users with < MIN_INTERACTIONS go entirely to train."""
    cached_train = _load("train_df")
    cached_test  = _load("test_df")
    if cached_train is not None and cached_test is not None:
        return cached_train, cached_test  # type: ignore[return-value]

    train_rows: list[pd.DataFrame] = []
    test_rows:  list[pd.DataFrame] = []
    sparse_users = 0
    split_users  = 0

    for _, group in df.groupby("user_id"):
        group = group.sort_values("report_date")
        if len(group) < MIN_INTERACTIONS:
            train_rows.append(group)
            sparse_users += 1
        else:
            n_test = max(1, int(len(group) * TEST_RATIO))
            train_rows.append(group.iloc[:-n_test])
            test_rows.append(group.iloc[-n_test:])
            split_users += 1

    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df  = pd.concat(test_rows).reset_index(drop=True) if test_rows else pd.DataFrame()

    logger.info(
        f"Split: {split_users} users split (train/test), "
        f"{sparse_users} sparse users (train only)"
    )
    logger.info(f"  Train: {len(train_df)} rows  |  Test: {len(test_df)} rows")
    _save("train_df", train_df)
    _save("test_df", test_df)
    return train_df, test_df


def evaluate(
    model: TwoTowerModel,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    index_maps: IndexMaps,
    video_embeddings_finetuned: dict[str, np.ndarray],
    k_values: list[int] = [10, 20, 50],
) -> None:
    device = next(model.parameters()).device
    model.eval()

    vid_ids_ordered = list(video_embeddings_finetuned.keys())
    vid_matrix = torch.tensor(
        np.stack([video_embeddings_finetuned[v] for v in vid_ids_ordered]),
        dtype=torch.float32,
    ).to(device)

    recalls:    dict[int, list[float]] = {k: [] for k in k_values}
    mrr_scores: list[float] = []
    cos_sims:   list[float] = []

    train_eng = compute_engagement_scores(train_df)
    train_ctx_map = (
        train_eng.groupby(["user_id", "brand_id"])
        .agg(
            watch_percentage=("watch_percentage", "mean"),
            views=("views", "mean"),
            likes=("likes", "mean"),
            shares=("shares", "mean"),
            comments=("comments", "mean"),
            link_clicks=("link_clicks", "mean"),
        )
        .to_dict("index")
    )

    with torch.no_grad():
        for (user_id, brand_id), group in test_df.groupby(["user_id", "brand_id"]):
            user_idx  = index_maps.get_user_idx(str(user_id), str(brand_id))
            brand_idx = index_maps.get_brand_idx(str(brand_id))

            ctx = train_ctx_map.get((user_id, brand_id), {})
            avg_eng = np.array([
                float(ctx.get("watch_percentage", 0.5)) / 100.0,
                min(float(ctx.get("views", 1.0)), 5.0) / 5.0,
                float(ctx.get("likes", 0.0)),
                float(ctx.get("shares", 0.0)),
                float(ctx.get("comments", 0.0)),
                float(ctx.get("link_clicks", 0.0)),
            ], dtype=np.float32)

            user_emb    = get_user_embedding(model, user_idx, brand_idx, avg_eng)
            user_tensor = torch.tensor(user_emb, dtype=torch.float32).unsqueeze(0).to(device)

            scores         = (user_tensor.expand(len(vid_ids_ordered), -1) * vid_matrix).sum(dim=-1) * model.temperature
            ranked_ids     = [vid_ids_ordered[i] for i in scores.argsort(descending=True).cpu().numpy()]

            test_positives = set(group["video_id"].tolist()) & set(vid_ids_ordered)
            if not test_positives:
                continue

            for k in k_values:
                recalls[k].append(len(test_positives & set(ranked_ids[:k])) / len(test_positives))

            for rank, vid_id in enumerate(ranked_ids, 1):
                if vid_id in test_positives:
                    mrr_scores.append(1.0 / rank)
                    break
            else:
                mrr_scores.append(0.0)

            pos_sims = [
                float(np.dot(user_emb, video_embeddings_finetuned[v]))
                for v in test_positives
                if video_embeddings_finetuned.get(v) is not None
            ]
            if not pos_sims:
                continue

            neg_ids = np.random.choice(
                [v for v in vid_ids_ordered if v not in test_positives],
                size=min(len(pos_sims), len(vid_ids_ordered) - len(test_positives)),
                replace=False,
            )
            neg_sims = [float(np.dot(user_emb, video_embeddings_finetuned[v])) for v in neg_ids]
            cos_sims.append(float(np.mean(pos_sims)) - float(np.mean(neg_sims)))

    n_videos = len(vid_ids_ordered)
    n_eval   = len(mrr_scores)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS (per-user chronological split)")
    print("=" * 60)
    print(f"  Users evaluated:  {n_eval}")
    print(f"  MRR:              {np.mean(mrr_scores):.4f}")
    for k in k_values:
        print(f"  Recall@{k:<3}:       {np.mean(recalls[k]):.4f}")
    print(f"\n  Cosine similarity gap (positive - negative videos):")
    print(f"    mean={np.mean(cos_sims):.4f}  median={np.median(cos_sims):.4f}  std={np.std(cos_sims):.4f}")
    print(f"    min={np.min(cos_sims):.4f}  max={np.max(cos_sims):.4f}")
    print("    (positive gap = model places user closer to correct videos)")
    print("\n  Random baseline:")
    for k in k_values:
        print(f"  Recall@{k:<3}:       {k / n_videos:.4f}")
    print("  Cosine gap (random): ~0.0000")


def get_all_user_embeddings(
    model: TwoTowerModel,
    train_df: pd.DataFrame,
    index_maps: IndexMaps,
) -> dict[str, np.ndarray]:
    cached = _load("user_embeddings_all")
    if cached is not None:
        return cached  # type: ignore[return-value]

    engagement_df = compute_engagement_scores(train_df)

    result: dict[str, np.ndarray] = {}
    for (user_id, brand_id), group in engagement_df.groupby(["user_id", "brand_id"]):
        uid = str(user_id)
        bid = str(brand_id)
        user_idx  = index_maps.get_user_idx(uid, bid)
        brand_idx = index_maps.get_brand_idx(bid)
        avg_eng   = np.array([
            float(group["watch_percentage"].mean()) / 100.0,
            min(float(group["views"].mean()), 5.0) / 5.0,
            float(group["likes"].mean()),
            float(group["shares"].mean()),
            float(group["comments"].mean()),
            float(group["link_clicks"].mean()) if "link_clicks" in group.columns else 0.0,
        ], dtype=np.float32)
        result[f"{uid}::{bid}"] = get_user_embedding(model, user_idx, brand_idx, avg_eng)

    _save("user_embeddings_all", result)
    logger.info(f"Computed embeddings for {len(result)} (user, brand) pairs")
    return result




def main():
    all_df = load_events()

    train_df, test_df = per_user_split(all_df)

    all_video_ids    = all_df["video_id"].dropna().unique().tolist()
    videos           = get_videos(all_video_ids)
    video_embeddings = get_video_embeddings_cached(videos)

    engagement_df = compute_engagement_scores(train_df)

    index_maps = IndexMaps(train_df)
    logger.info(f"Index maps: {index_maps.n_users} users, {index_maps.n_brands} brands")

    logger.info("Building training dataset...")
    dataset = EngagementDataset(
        engagement_df=engagement_df,
        index_maps=index_maps,
        video_embeddings=video_embeddings,
        positive_threshold=0.05,
        negative_ratio=4,
        hard_negative_ratio=0,
    )

    logger.info("Training two-tower model...")
    model = TwoTowerModel(
        n_users=index_maps.n_users,
        n_brands=index_maps.n_brands,
        backbone_dim=768,
        output_dim=512,
        temperature=10.0,
    )
    losses = train(model, dataset, epochs=100, batch_size=256, lr=1e-3)
    logger.info(f"Training complete. Final loss: {losses[-1]:.4f}")

    logger.info("Projecting video embeddings through finetuned video tower...")
    finetuned_videos = get_video_embeddings_finetuned(model, video_embeddings)

    evaluate(
        model=model,
        train_df=train_df,
        test_df=test_df,
        index_maps=index_maps,
        video_embeddings_finetuned=finetuned_videos,
        k_values=[10, 20, 50],
    )

    logger.info("Computing user embeddings for all training users...")
    all_user_embs = get_all_user_embeddings(model, train_df, index_maps)

    logger.info("Clustering videos into content cohorts...")
    video_cohort_map, km = cluster_videos(finetuned_videos)
    _save("video_cohort_map", video_cohort_map)

    logger.info("Assigning users to cohorts based on content engagement...")
    user_cohort_map = assign_user_cohorts(engagement_df, video_cohort_map)
    _save("user_cohort_map", user_cohort_map)
    multi = sum(1 for v in user_cohort_map.values() if len(v) > 1)
    logger.info(f"  {len(user_cohort_map)} users assigned; {multi} belong to multiple cohorts")

    logger.info("Building cohort profiles...")
    cohort_profiles = build_cohort_profiles(video_cohort_map, engagement_df, videos, k=km.n_clusters)
    logger.info("Generating cohort labels via Claude...")
    cohort_profiles = generate_all_labels(cohort_profiles, video_cohort_map, videos)
    logger.info(f"Built {len(cohort_profiles)} cohort profiles")
    _save("cohort_profiles", cohort_profiles)

    logger.info("Saving model checkpoint...")
    torch.save(model.state_dict(), CACHE_DIR / "two_tower.pt")
    logger.info(f"Model saved → {CACHE_DIR / 'two_tower.pt'}")

    logger.info("Dumping embeddings to Milvus...")
    video_brand_map = all_df.groupby("video_id")["brand_id"].first().to_dict()
    upsert_video_embeddings(finetuned_videos, video_brand_map)
    upsert_user_embeddings(all_user_embs, user_cohort_map)


main()


# import asyncio
# import logging
# import pandas as pd
# from db.clickhouse import get_user_data_by_brand_id

# logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
# logger = logging.getLogger(__name__)

# BRAND_IDS = "(1729, 2023, 2075, 2556, 2558, 2314, 2357, 2023, 2476, 2557, 2701, 2764, 2790, 2793, 2801, 2808, 3099)"
# # NUM_ROWS = 28051226
# NUM_ROWS = 539951
# DATE_START  = "2026-01-01"
# DATE_END    = "2026-04-29"
# BATCH_SIZE = 50000
# FILE_NAME = 'ck_user_events.csv'

# async def get_all_events() -> pd.DataFrame:
#     # cached = _load("all_events")
#     # if cached is not None:
#     #     return cached  # type: ignore[return-value]
#     all_df = pd.read_csv(FILE_NAME)
#     existing_rows = len(all_df)
#     logger.info(f"Existing rows in CSV: {existing_rows}")

#     for i in range(existing_rows, NUM_ROWS, BATCH_SIZE):
#         logger.info(f"Fetching events {i} to {i + BATCH_SIZE}...")
#         df = await get_user_data_by_brand_id(
#             brand_ids=BRAND_IDS,
#             start_date=DATE_START,
#             end_date=DATE_END,
#             limit=BATCH_SIZE,
#             offset=i,
#         )
#         if i == 0:
#             all_df = df
#         else:
#             all_df = pd.concat([all_df, df], ignore_index=True)
#         all_df.to_csv(FILE_NAME, index=False)
#     # _save("all_events", df)
#     return all_df


# async def main():
#     all_df = await get_all_events()
#     all_df.to_csv(FILE_NAME, index=False)


# asyncio.run(main())