"""
Fetches user event data from ClickHouse and saves to CSV.

Applies qualified_users filter (> 10 events per user per brand) and
fetches in batches of 50,000 rows. Resumes from where it left off if
the CSV already exists (safe to re-run after interruption).

Usage:
    uv run fetch_data.py
"""

import asyncio
import logging
import os
from pathlib import Path

import pandas as pd

from db.clickhouse import get_user_data_by_brand_id

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BRAND_IDS  = "(1729, 2023, 2075, 2556, 2558, 2314, 2357, 2476, 2557, 2701, 2764, 2790, 2793, 2801, 2808, 3099)"
DATE_START = "2025-05-01"
DATE_END   = "2026-05-19"
BATCH_SIZE = 50_000
MIN_EVENTS = 10
OUT_FILE   = "ck_user_events.csv"


async def fetch() -> None:
    out = Path(OUT_FILE)

    # Resume from existing rows if file already exists
    if out.exists():
        existing = pd.read_csv(out)
        start_offset = len(existing)
        logger.info(f"Resuming from offset {start_offset} (existing rows: {start_offset})")
    else:
        existing = pd.DataFrame()
        start_offset = 0

    all_frames: list[pd.DataFrame] = [existing] if not existing.empty else []
    offset = start_offset
    total_fetched = 0

    while True:
        logger.info(f"Fetching batch at offset {offset}...")
        batch = await get_user_data_by_brand_id(
            brand_ids=BRAND_IDS,
            start_date=DATE_START,
            end_date=DATE_END,
            min_events=MIN_EVENTS,
            limit=BATCH_SIZE,
            offset=offset,
        )

        if batch.empty:
            logger.info("No more rows — fetch complete.")
            break

        # Strip stray quotes from IDs
        batch["user_id"]  = batch["user_id"].astype(str).str.strip('"')
        batch["video_id"] = batch["video_id"].astype(str).str.strip('"')

        all_frames.append(batch)
        total_fetched += len(batch)
        offset += len(batch)

        # Save after every batch so we can resume on interruption
        combined = pd.concat(all_frames, ignore_index=True)
        combined.to_csv(out, index=False)
        logger.info(f"  Saved {len(combined)} total rows → {out}")

        if len(batch) < BATCH_SIZE:
            logger.info("Last batch was smaller than batch size — fetch complete.")
            break

    final = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    final.to_csv(out, index=False)
    logger.info(f"Done. {len(final)} rows, {final['user_id'].nunique() if not final.empty else 0} unique users → {out}")


asyncio.run(fetch())
