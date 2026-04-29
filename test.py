import asyncio
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from db.clickhouse import get_user_data_by_brand_id
from db.opensearch import get_videos_by_ids
from utils.engagement import compute_engagement_scores
from utils.embeddings import embed_videos
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

BRAND_ID    = 2357
DATE_START  = "2026-01-01"
DATE_END    = "2026-04-28"
MIN_INTERACTIONS = 5    # users with fewer go entirely to train
TEST_RATIO       = 0.2  # most recent 20% of each user's history → test


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


async def get_all_events() -> pd.DataFrame:
    cached = _load("all_events")
    if cached is not None:
        return cached  # type: ignore[return-value]
    df = await get_user_data_by_brand_id(
        brand_id=BRAND_ID, start_date=DATE_START, end_date=DATE_END
    )
    _save("all_events", df)
    return df


async def get_videos(video_ids: list[str]) -> list[dict]:
    cached = _load("videos")
    if cached is not None:
        return cached  # type: ignore[return-value]
    logger.info(f"Fetching {len(video_ids)} videos from OpenSearch...")
    videos = await get_videos_by_ids(video_ids)
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
    """
    Per-user chronological split:
      - Users with >= MIN_INTERACTIONS: oldest 80% → train, newest 20% → test
      - Users with < MIN_INTERACTIONS: all → train, none → test

    Chronological order within each user prevents data leakage.
    """
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
    return train_df, test_df


def evaluate(
    model: TwoTowerModel,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    index_maps: IndexMaps,
    video_embeddings_finetuned: dict[str, np.ndarray],
    k_values: list[int] = [10, 20, 50],
) -> None:
    """
    Per-user evaluation on held-out test interactions.

    Two metrics:
    1. Recall@K / MRR  — ranking quality
    2. Cosine similarity — how close the predicted user embedding
                          is to the actual test video embeddings
    """
    device = next(model.parameters()).device
    model.eval()

    vid_ids_ordered = list(video_embeddings_finetuned.keys())
    vid_matrix = torch.tensor(
        np.stack([video_embeddings_finetuned[v] for v in vid_ids_ordered]),
        dtype=torch.float32
    ).to(device)

    recalls:    dict[int, list[float]] = {k: [] for k in k_values}
    mrr_scores: list[float] = []
    cos_sims:   list[float] = []

    # Build per-user train context for engagement features
    train_eng = compute_engagement_scores(train_df)
    train_ctx = train_eng.groupby("user_id").agg(
        watch_percentage=("watch_percentage", "mean"),
        views=("views", "mean"),
        likes=("likes", "mean"),
        shares=("shares", "mean"),
        comments=("comments", "mean"),
    ).reset_index()
    train_ctx_map = train_ctx.set_index("user_id").to_dict("index")

    with torch.no_grad():
        for user_id, group in test_df.groupby("user_id"):
            user_idx  = index_maps.get_user_idx(str(user_id))
            brand_idx = index_maps.get_brand_idx(str(group["brand_id"].iloc[0]))

            # Use train-time engagement as user context (not test leakage)
            ctx = train_ctx_map.get(user_id, {})
            avg_eng = np.array([
                float(ctx.get("watch_percentage", 0.5)) / 100.0,
                min(float(ctx.get("views", 1.0)), 5.0) / 5.0,
                float(ctx.get("likes", 0.0)),
                float(ctx.get("shares", 0.0)),
                float(ctx.get("comments", 0.0)),
            ], dtype=np.float32)

            user_emb   = get_user_embedding(model, user_idx, brand_idx, avg_eng)
            user_tensor = torch.tensor(user_emb, dtype=torch.float32).unsqueeze(0).to(device)

            # Rank all videos
            user_expanded  = user_tensor.expand(len(vid_ids_ordered), -1)
            scores         = (user_expanded * vid_matrix).sum(dim=-1) * model.temperature
            ranked_indices = scores.argsort(descending=True).cpu().numpy()
            ranked_ids     = [vid_ids_ordered[i] for i in ranked_indices]

            test_positives = set(group["video_id"].tolist()) & set(vid_ids_ordered)
            if not test_positives:
                continue

            # Recall@K
            for k in k_values:
                top_k = set(ranked_ids[:k])
                recalls[k].append(len(test_positives & top_k) / len(test_positives))

            # MRR
            for rank, vid_id in enumerate(ranked_ids, 1):
                if vid_id in test_positives:
                    mrr_scores.append(1.0 / rank)
                    break
            else:
                mrr_scores.append(0.0)

            # Cosine similarity gap: positive videos vs random negatives
            # Measures whether model places user closer to correct videos
            # than to random ones — sign and magnitude both matter
            pos_sims = []
            for vid_id in test_positives:
                vid_emb = video_embeddings_finetuned.get(vid_id)
                if vid_emb is None:
                    continue
                pos_sims.append(float(np.dot(user_emb, vid_emb)))

            if not pos_sims:
                continue

            # Sample same number of random negatives for fair comparison
            neg_ids = np.random.choice(
                [v for v in vid_ids_ordered if v not in test_positives],
                size=min(len(pos_sims), len(vid_ids_ordered) - len(test_positives)),
                replace=False,
            )
            neg_sims = [
                float(np.dot(user_emb, video_embeddings_finetuned[v]))
                for v in neg_ids
            ]

            gap = np.mean(pos_sims) - np.mean(neg_sims)
            cos_sims.append(gap)

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
    print(f"    mean={np.mean(cos_sims):.4f}  "
          f"median={np.median(cos_sims):.4f}  "
          f"std={np.std(cos_sims):.4f}")
    print(f"    min={np.min(cos_sims):.4f}  max={np.max(cos_sims):.4f}")
    print("    (positive gap = model places user closer to correct videos)")
    print("\n  Random baseline:")
    for k in k_values:
        print(f"  Recall@{k:<3}:       {k/n_videos:.4f}")
    print("  Cosine gap (random): ~0.0000")


async def main():
    # Fetch all events in one query — split is done in Python
    logger.info(f"Fetching all events ({DATE_START} to {DATE_END})...")
    all_df = await get_all_events()
    logger.info(f"  {len(all_df)} rows, {all_df['user_id'].nunique()} unique users")

    # Per-user chronological split
    train_df, test_df = per_user_split(all_df)

    # Videos from full dataset
    all_video_ids = all_df["video_id"].dropna().unique().tolist()
    videos = await get_videos(all_video_ids)
    video_embeddings = get_video_embeddings_cached(videos)

    # Engagement scores on train only
    engagement_df = compute_engagement_scores(train_df)

    # Index maps built from train only
    index_maps = IndexMaps(train_df)
    logger.info(f"Index maps: {index_maps.n_users} users, {index_maps.n_brands} brands")

    # Dataset
    logger.info("Building training dataset...")
    dataset = EngagementDataset(
        engagement_df=engagement_df,
        index_maps=index_maps,
        video_embeddings=video_embeddings,
        positive_threshold=0.05,
        negative_ratio=4,
    )

    # Train
    logger.info("Training two-tower model...")
    model = TwoTowerModel(
        n_users=index_maps.n_users,
        n_brands=index_maps.n_brands,
        backbone_dim=384,
        output_dim=128,
        temperature=10.0,
    )
    losses = train(model, dataset, epochs=50, batch_size=256, lr=1e-3)
    logger.info(f"Training complete. Final loss: {losses[-1]:.4f}")

    # Finetuned video embeddings
    logger.info("Projecting video embeddings through finetuned video tower...")
    finetuned_videos = get_video_embeddings_finetuned(model, video_embeddings)

    # Evaluate
    evaluate(
        model=model,
        train_df=train_df,
        test_df=test_df,
        index_maps=index_maps,
        video_embeddings_finetuned=finetuned_videos,
        k_values=[10, 20, 50],
    )


asyncio.run(main())
