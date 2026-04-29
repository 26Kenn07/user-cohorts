import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


def find_optimal_k(embeddings: np.ndarray, k_min: int = 3, k_max: int = 8) -> int:
    """
    Finds optimal number of clusters using silhouette score.
    With small datasets we cap at k_max to avoid over-clustering.
    """
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


def cluster_users(
    user_embeddings: dict[str, np.ndarray],
    k: int | None = None,
) -> tuple[dict[str, int], KMeans]:
    """
    Clusters user embeddings into k cohorts.
    If k is None, finds optimal k automatically.
    Returns (user_id -> cohort_id mapping, fitted KMeans).
    """
    user_ids = list(user_embeddings.keys())
    matrix = np.stack([user_embeddings[uid] for uid in user_ids])

    if k is None:
        logger.info("Finding optimal number of cohorts...")
        k = find_optimal_k(matrix)

    logger.info(f"Clustering {len(user_ids)} users into {k} cohorts...")
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(matrix)

    cohort_map = {uid: int(labels[i]) for i, uid in enumerate(user_ids)}

    # Log cohort sizes
    for cohort_id in range(k):
        size = sum(1 for c in cohort_map.values() if c == cohort_id)
        logger.info(f"  Cohort {cohort_id}: {size} users")

    return cohort_map, km


def get_cohort_top_keywords(
    cohort_id: int,
    cohort_map: dict[str, int],
    engagement_df: pd.DataFrame,
    videos: list[dict[str, Any]],
    top_n: int = 15,
) -> list[str]:
    """
    For a given cohort, finds the most common keywords across
    videos that users in that cohort engaged with.
    """
    cohort_users = {uid for uid, cid in cohort_map.items() if cid == cohort_id}
    cohort_events = engagement_df[engagement_df["user_id"].isin(cohort_users)]

    # Weight keyword frequency by engagement score
    video_lookup = {v["video_id"]: v for v in videos}
    keyword_scores: dict[str, float] = {}

    for _, row in cohort_events.iterrows():
        video = video_lookup.get(row["video_id"])
        if not video:
            continue
        keywords = video.get("keywords") or []
        for kw in keywords:
            keyword_scores[kw] = keyword_scores.get(kw, 0.0) + row["score"]

    sorted_kws = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
    return [kw for kw, _ in sorted_kws[:top_n]]


def build_cohort_profiles(
    cohort_map: dict[str, int],
    engagement_df: pd.DataFrame,
    videos: list[dict[str, Any]],
    k: int,
) -> list[dict[str, Any]]:
    """
    Builds a profile for each cohort containing top keywords and stats.
    """
    profiles = []
    for cohort_id in range(k):
        top_keywords = get_cohort_top_keywords(
            cohort_id, cohort_map, engagement_df, videos
        )
        cohort_users = [uid for uid, cid in cohort_map.items() if cid == cohort_id]
        profiles.append({
            "cohort_id": cohort_id,
            "user_count": len(cohort_users),
            "top_keywords": top_keywords,
        })
        logger.info(f"Cohort {cohort_id} ({len(cohort_users)} users) — top keywords: {top_keywords[:5]}")

    return profiles
