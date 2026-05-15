import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


def find_optimal_k(embeddings: np.ndarray, k_min: int = 3, k_max: int = 8) -> int:
    best_k = k_min
    best_score = -1.0

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        logger.info(f"  k={k} silhouette={score:.4f}")
        if score > best_score:
            best_score = score
            best_k = k

    logger.info(f"Optimal k={best_k} (silhouette={best_score:.4f})")
    return best_k


def cluster_videos(
    video_embeddings: dict[str, np.ndarray],
    k: int | None = None,
) -> tuple[dict[str, int], KMeans]:
    """
    Clusters videos by finetuned content embeddings.
    Cohorts reflect content topics (entertainment, wellness, automotive, etc.)
    rather than brand geography.
    Returns (video_id → cohort_id, fitted KMeans).
    """
    video_ids = list(video_embeddings.keys())
    matrix = np.stack([video_embeddings[vid] for vid in video_ids])

    if k is None:
        logger.info("Finding optimal number of video cohorts...")
        k = find_optimal_k(matrix)

    logger.info(f"Clustering {len(video_ids)} videos into {k} cohorts...")
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(matrix)

    video_cohort_map = {vid: int(labels[i]) for i, vid in enumerate(video_ids)}

    for cohort_id in range(k):
        size = sum(1 for c in video_cohort_map.values() if c == cohort_id)
        logger.info(f"  Video cohort {cohort_id}: {size} videos")

    return video_cohort_map, km


def assign_user_cohorts(
    engagement_df: pd.DataFrame,
    video_cohort_map: dict[str, int],
    min_engagement_share: float = 0.15,
) -> dict[str, list[int]]:
    """
    Assigns each (user, brand) pair to one or more cohorts based on content engagement.

    A user is assigned to a cohort if that cohort accounts for >= min_engagement_share
    of their total engagement score. Users with diverse interests get multiple cohorts.
    Falls back to the single highest-scoring cohort if none clears the threshold.

    Returns user_brand_key ("user_id::brand_id") → sorted list of cohort IDs.
    """
    result: dict[str, list[int]] = {}

    for (user_id, brand_id), group in engagement_df.groupby(["user_id", "brand_id"]):
        key = f"{user_id}::{brand_id}"

        cohort_scores: dict[int, float] = {}
        total_score = 0.0

        for _, row in group.iterrows():
            cohort_id = video_cohort_map.get(str(row["video_id"]))
            if cohort_id is None:
                continue
            score = float(row["score"])
            cohort_scores[cohort_id] = cohort_scores.get(cohort_id, 0.0) + score
            total_score += score

        if not cohort_scores:
            result[key] = [0]
            continue

        if total_score == 0.0:
            result[key] = [min(cohort_scores)]
            continue

        assigned = sorted(
            cid for cid, s in cohort_scores.items()
            if s / total_score >= min_engagement_share
        )
        # Always assign at least the dominant cohort
        result[key] = assigned if assigned else [max(cohort_scores, key=cohort_scores.get)]

    n_multi = sum(1 for v in result.values() if len(v) > 1)
    logger.info(f"User cohort assignment: {len(result)} users, {n_multi} multi-cohort")
    return result


def get_cohort_top_keywords(
    cohort_id: int,
    video_cohort_map: dict[str, int],
    videos: list[dict[str, Any]],
    engagement_df: pd.DataFrame,
    top_n: int = 15,
) -> list[str]:
    """
    Returns the top keywords for a video cohort, weighted by engagement score
    across all users who watched videos in that cluster.
    """
    cohort_video_ids = {vid for vid, cid in video_cohort_map.items() if cid == cohort_id}
    cohort_events = engagement_df[engagement_df["video_id"].isin(cohort_video_ids)]

    video_lookup = {v["video_id"]: v for v in videos}
    keyword_scores: dict[str, float] = {}

    for _, row in cohort_events.iterrows():
        video = video_lookup.get(row["video_id"])
        if not video:
            continue
        for kw in (video.get("keywords") or []):
            keyword_scores[kw] = keyword_scores.get(kw, 0.0) + float(row["score"])

    sorted_kws = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
    return [kw for kw, _ in sorted_kws[:top_n]]


def build_cohort_profiles(
    video_cohort_map: dict[str, int],
    engagement_df: pd.DataFrame,
    videos: list[dict[str, Any]],
    k: int,
) -> list[dict[str, Any]]:
    profiles = []
    for cohort_id in range(k):
        top_keywords = get_cohort_top_keywords(
            cohort_id, video_cohort_map, videos, engagement_df
        )
        n_videos = sum(1 for cid in video_cohort_map.values() if cid == cohort_id)
        profiles.append({
            "cohort_id":    cohort_id,
            "video_count":  n_videos,
            "top_keywords": top_keywords,
        })
        logger.info(f"Cohort {cohort_id} ({n_videos} videos) — top keywords: {top_keywords[:5]}")

    return profiles
