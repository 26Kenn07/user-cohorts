"""
Page-context pipeline for web-only users (ck_user_events_with_url.csv).
Trains WITH page context — compare results against cache_web_no_ctx/eval_metrics.pkl.

Isolated from all other pipelines:
  - cache dir  : cache_web_ctx/
  - model file : cache_web_ctx/two_tower_web_ctx.pt
  - Milvus     : video_embeddings_web_ctx / user_embeddings_web_ctx
"""
import ast
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from db.milvus import drop_collection, upsert_video_embeddings, upsert_user_embeddings
from utils.engagement import compute_engagement_scores
from utils.embeddings import embed_videos
from utils.page_context import extract_page_context, embed_page_contexts, get_user_page_ctx_embs
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

CACHE_DIR        = Path("cache_web_ctx")
CSV_FILE         = "ck_user_events_with_url.csv"
VIDEO_CSV_FILE   = "new_video_data.csv"
METADATA_FILE    = "combined_metadata.json"
MODEL_FILE       = CACHE_DIR / "two_tower_web_ctx.pt"
VIDEO_COLLECTION = "video_embeddings_web_ctx"
USER_COLLECTION  = "user_embeddings_web_ctx"
MIN_INTERACTIONS = 10
TEST_RATIO       = 0.2

CACHE_DIR.mkdir(exist_ok=True)


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
    df["user_id"] = df["user_id"].astype(str).str.strip('"')
    df["video_id"] = df["video_id"].astype(str).str.strip('"')
    # Preserve real NaN for missing URLs — don't stringify to "nan"
    if "url" in df.columns:
        df["url"] = df["url"].astype(str).str.strip()
        df["url"] = df["url"].replace({"nan": pd.NA, "": pd.NA, "None": pd.NA, "none": pd.NA})
    else:
        df["url"] = pd.NA
    n_with_url = df["url"].notna().sum()
    logger.info(f"  {len(df)} rows, {df['user_id'].nunique()} users — "
                f"{n_with_url} rows have URL ({len(df) - n_with_url} without)")
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


def load_page_ctx_embs(events_df: pd.DataFrame) -> dict[str, np.ndarray]:
    cached = _load("user_page_ctx_embs")
    if cached is not None:
        return cached  # type: ignore[return-value]

    # Load metadata file
    logger.info(f"Loading page metadata from {METADATA_FILE}...")
    with open(METADATA_FILE) as f:
        raw_metadata: dict = json.load(f)

    # Extract context strings for every unique URL in events
    unique_urls = events_df["url"].dropna().astype(str).unique().tolist()
    logger.info(f"Extracting context for {len(unique_urls)} unique URLs...")
    url_context_map: dict[str, str] = {}
    missing_meta = 0
    for url in unique_urls:
        entry = raw_metadata.get(url)
        if entry:
            url_context_map[url] = extract_page_context(entry)
        else:
            url_context_map[url] = ""
            missing_meta += 1
    if missing_meta:
        logger.warning(f"{missing_meta}/{len(unique_urls)} URLs have no metadata entry — using empty context")

    # Embed (cached separately so we don't re-embed if only user map changes)
    url_emb_cached = _load("url_embeddings")
    if url_emb_cached is not None:
        url_emb_map: dict[str, np.ndarray] = url_emb_cached  # type: ignore[assignment]
    else:
        url_emb_map = embed_page_contexts(url_context_map)
        _save("url_embeddings", url_emb_map)

    # Average per (user, brand)
    user_page_ctx_embs = get_user_page_ctx_embs(events_df, url_emb_map)
    _save("user_page_ctx_embs", user_page_ctx_embs)
    return user_page_ctx_embs


def per_user_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    user_page_ctx_embs: dict[str, np.ndarray],
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

            page_ctx = user_page_ctx_embs.get(f"{user_id}::{brand_id}")
            user_emb = get_user_embedding(model, user_idx, brand_idx, avg_eng, page_ctx)
            user_tensor = torch.tensor(user_emb, dtype=torch.float32).unsqueeze(0).to(device)

            scores     = (user_tensor.expand(len(vid_ids_ordered), -1) * vid_matrix).sum(dim=-1) * model.temperature
            ranked_ids = [vid_ids_ordered[i] for i in scores.argsort(descending=True).cpu().numpy()]

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
    print("EVALUATION RESULTS — web-only WITH page context")
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

    # Print comparison with no-context baseline if available
    baseline_path = Path("cache_web_no_ctx/eval_metrics.pkl")
    if baseline_path.exists():
        with open(baseline_path, "rb") as f:
            baseline: dict = pickle.load(f)
        print("\n" + "=" * 60)
        print("COMPARISON vs. no-context baseline")
        print("=" * 60)
        mrr_delta = np.mean(mrr_scores) - baseline["mrr"]
        print(f"  MRR:       {baseline['mrr']:.4f} → {np.mean(mrr_scores):.4f}  ({mrr_delta:+.4f})")
        for k in k_values:
            base_r = baseline.get(f"recall@{k}", 0.0)
            curr_r = np.mean(recalls[k])
            print(f"  Recall@{k:<3}: {base_r:.4f} → {curr_r:.4f}  ({curr_r - base_r:+.4f})")

    metrics = {
        "variant": "web_ctx",
        "n_eval": n_eval,
        "mrr": float(np.mean(mrr_scores)),
        **{f"recall@{k}": float(np.mean(recalls[k])) for k in k_values},
        "cosine_gap_mean": float(np.mean(cos_sims)),
        "cosine_gap_median": float(np.median(cos_sims)),
    }
    _save("eval_metrics", metrics)
    logger.info("Saved eval metrics → cache_web_ctx/eval_metrics.pkl")


def get_all_user_embeddings(
    model: TwoTowerModel,
    train_df: pd.DataFrame,
    index_maps: IndexMaps,
    user_page_ctx_embs: dict[str, np.ndarray],
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
        page_ctx = user_page_ctx_embs.get(f"{uid}::{bid}")
        result[f"{uid}::{bid}"] = get_user_embedding(model, user_idx, brand_idx, avg_eng, page_ctx)

    _save("user_embeddings_all", result)
    logger.info(f"Computed embeddings for {len(result)} (user, brand) pairs")
    return result


def main():
    logger.info("Dropping stale Milvus collections before training...")
    drop_collection(VIDEO_COLLECTION)
    drop_collection(USER_COLLECTION)

    all_df = load_events()

    train_df, test_df = per_user_split(all_df)

    all_video_ids    = all_df["video_id"].dropna().unique().tolist()
    videos           = get_videos(all_video_ids)
    video_embeddings = get_video_embeddings_cached(videos)

    # Page context: embed URLs, average per (user, brand)
    user_page_ctx_embs = load_page_ctx_embs(all_df)

    engagement_df = compute_engagement_scores(train_df)
    index_maps    = IndexMaps(train_df)
    logger.info(f"Index maps: {index_maps.n_users} users, {index_maps.n_brands} brands")

    logger.info("Building training dataset with page context...")
    dataset = EngagementDataset(
        engagement_df=engagement_df,
        index_maps=index_maps,
        video_embeddings=video_embeddings,
        positive_threshold=0.05,
        negative_ratio=4,
        hard_negative_ratio=0,
        user_page_ctx_map=user_page_ctx_embs,
    )

    logger.info("Training two-tower model (web-only, WITH page context)...")
    model = TwoTowerModel(
        n_users=index_maps.n_users,
        n_brands=index_maps.n_brands,
        backbone_dim=768,
        output_dim=512,
        temperature=10.0,
        use_page_ctx=True,
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
        user_page_ctx_embs=user_page_ctx_embs,
        k_values=[10, 20, 50],
    )

    logger.info("Computing user embeddings for all training users...")
    all_user_embs = get_all_user_embeddings(model, train_df, index_maps, user_page_ctx_embs)

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

    logger.info(f"Saving model checkpoint → {MODEL_FILE}")
    torch.save(model.state_dict(), MODEL_FILE)

    logger.info("Dumping embeddings to Milvus (isolated collections)...")
    video_brand_map = all_df.groupby("video_id")["brand_id"].first().to_dict()
    upsert_video_embeddings(finetuned_videos, video_brand_map, collection=VIDEO_COLLECTION)
    upsert_user_embeddings(all_user_embs, user_cohort_map, collection=USER_COLLECTION)


main()
