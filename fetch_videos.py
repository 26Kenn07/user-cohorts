"""
Fetches video metadata from OpenSearch for all unique video_ids in ck_user_events.csv
and saves to new_video_data.csv.

Usage:
    uv run fetch_videos.py
    uv run fetch_videos.py --input ck_user_events.csv --output new_video_data.csv
"""

import argparse
import asyncio
import logging
from pathlib import Path

import pandas as pd

from db.opensearch import get_videos_by_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _parse_keywords(raw: object) -> str:
    """Stores keywords as a Python list repr so test.py can ast.literal_eval it back."""
    if isinstance(raw, list):
        return repr([str(k) for k in raw if k])
    return repr([])


def _confidence(video: dict) -> str:
    filled = sum(1 for f in ["transcript", "description_text", "video_gen_description", "keywords"]
                 if video.get(f))
    if filled >= 3: return "high"
    if filled == 2: return "medium"
    if filled == 1: return "low"
    return "none"


async def fetch(input_csv: str, output_csv: str) -> None:
    out = Path(output_csv)

    # Load existing output to allow resuming
    if out.exists():
        existing = pd.read_csv(out)
        already_fetched = set(existing["video_id"].astype(str).tolist())
        logger.info(f"Resuming: {len(already_fetched)} videos already in {output_csv}")
    else:
        existing = pd.DataFrame()
        already_fetched = set()

    # Get unique video_ids from events CSV
    logger.info(f"Reading video_ids from {input_csv}...")
    events_df = pd.read_csv(input_csv, usecols=["video_id"])
    events_df["video_id"] = events_df["video_id"].astype(str).str.strip('"')
    all_video_ids = events_df["video_id"].dropna().unique().tolist()
    logger.info(f"  {len(all_video_ids)} unique video_ids in events")

    missing = [v for v in all_video_ids if v not in already_fetched]
    logger.info(f"  {len(missing)} videos to fetch from OpenSearch")

    if not missing:
        logger.info("Nothing to fetch — output is already up to date.")
        return

    # Fetch in batches of 500
    BATCH = 500
    all_rows = existing.to_dict("records") if not existing.empty else []
    n_batches = (len(missing) + BATCH - 1) // BATCH

    for i in range(0, len(missing), BATCH):
        batch_ids = missing[i : i + BATCH]
        batch_num = i // BATCH + 1
        logger.info(f"Fetching batch {batch_num}/{n_batches} ({len(batch_ids)} videos)...")

        videos = await get_videos_by_ids(batch_ids)

        fetched_ids = {v["video_id"] for v in videos}
        for vid_id in batch_ids:
            if vid_id not in fetched_ids:
                # Not found in OpenSearch — store empty row so we don't re-fetch
                all_rows.append({
                    "video_id": vid_id,
                    "transcript": "",
                    "description_text": "",
                    "video_gen_description": "",
                    "keywords": repr([]),
                    "embedding_confidence": "none",
                })

        for v in videos:
            all_rows.append({
                "video_id":             v["video_id"],
                "transcript":           v.get("transcript") or "",
                "description_text":     v.get("description_text") or "",
                "video_gen_description": v.get("video_gen_description") or "",
                "keywords":             _parse_keywords(v.get("keywords")),
                "embedding_confidence": _confidence(v),
            })

        # Save after each batch so it's safe to interrupt
        pd.DataFrame(all_rows).to_csv(out, index=False)
        logger.info(f"  Saved {len(all_rows)} total rows → {out}")

    logger.info(f"Done. {len(all_rows)} videos written to {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="ck_user_events.csv")
    parser.add_argument("--output", default="new_video_data.csv")
    args = parser.parse_args()
    asyncio.run(fetch(args.input, args.output))


main()
