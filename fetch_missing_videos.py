"""
Fetches metadata for video IDs that are in ck_user_events_with_url.csv
but missing from new_video_data.csv, then appends them to the CSV.

Usage:
    uv run fetch_missing_videos.py
"""

import asyncio
import logging

import pandas as pd

from db.opensearch import get_videos_by_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EVENTS_CSV = "ck_user_events_with_url.csv"
VIDEO_CSV  = "new_video_data.csv"


async def fetch() -> None:
    event_ids = set(pd.read_csv(EVENTS_CSV)["video_id"].astype(str).unique())
    video_ids = set(pd.read_csv(VIDEO_CSV)["video_id"].astype(str).unique())
    missing   = sorted(event_ids - video_ids)

    logger.info(f"Event videos : {len(event_ids)}")
    logger.info(f"Video CSV    : {len(video_ids)}")
    logger.info(f"Missing      : {len(missing)}")

    if not missing:
        logger.info("Nothing to fetch.")
        return

    videos = await get_videos_by_ids(missing)
    logger.info(f"Fetched {len(videos)} videos from OpenSearch")

    rows = [
        {
            "video_id":              v["video_id"],
            "transcript":            v.get("transcript") or "",
            "description_text":      v.get("description_text") or "",
            "video_gen_description": v.get("video_gen_description") or "",
            # Store as string repr of list to match existing CSV format
            "keywords":              str(v.get("keywords") or []),
            "embedding_confidence":  v.get("embedding_confidence") or "none",
        }
        for v in videos
    ]

    new_df = pd.DataFrame(rows)
    new_df.to_csv(VIDEO_CSV, mode="a", header=False, index=False)
    logger.info(f"Appended {len(new_df)} rows → {VIDEO_CSV}")

    still_missing = len(missing) - len(videos)
    if still_missing:
        logger.warning(f"{still_missing} IDs not found in OpenSearch (videos may have been deleted)")


asyncio.run(fetch())
