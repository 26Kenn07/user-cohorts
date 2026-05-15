"""
Show top keywords for each video-based cohort without re-running the pipeline.
Reads video_cohort_map, engagement data, and videos from cache.

Usage:
    uv run cohort_keywords.py
"""

import logging
import pickle
from pathlib import Path

from utils.cohort import get_cohort_top_keywords
from utils.engagement import compute_engagement_scores

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = Path("cache")


def load(name: str):
    with open(CACHE_DIR / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def main() -> None:
    logger.info("Loading video cohort map, engagement data, and videos from cache...")
    video_cohort_map: dict[str, int] = load("video_cohort_map")
    train_df    = load("train_df")
    videos      = load("videos")
    engagement_df = compute_engagement_scores(train_df)

    k = max(video_cohort_map.values()) + 1
    logger.info(f"Found {k} video cohorts")

    user_cohort_map: dict[str, list[int]] = load("user_cohort_map")
    n_multi = sum(1 for v in user_cohort_map.values() if len(v) > 1)
    logger.info(f"{len(user_cohort_map)} users assigned; {n_multi} belong to multiple cohorts")

    cohort_profiles: list[dict] = load("cohort_profiles")
    profile_map = {p["cohort_id"]: p for p in cohort_profiles}

    print("\n" + "=" * 60)
    for cohort_id in range(k):
        n_videos  = sum(1 for cid in video_cohort_map.values() if cid == cohort_id)
        n_users   = sum(1 for cohorts in user_cohort_map.values() if cohort_id in cohorts)
        keywords  = get_cohort_top_keywords(cohort_id, video_cohort_map, videos, engagement_df, top_n=15)
        profile   = profile_map.get(cohort_id, {})
        label     = profile.get("label", f"Cohort {cohort_id}")
        desc      = profile.get("description", "")
        print(f"\nCohort {cohort_id}  —  {label}")
        if desc:
            print(f"  {desc}")
        print(f"  ({n_videos:,} videos  |  {n_users:,} users)")
        print(f"  Top keywords: {', '.join(keywords)}")
    print("=" * 60)


main()
