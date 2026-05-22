"""
Finds iHear videos in OpenSearch that are not yet in Milvus,
embeds them via backbone + VideoTower, and upserts with has_engagement=False.

Usage:
    uv run ingest_missing_videos.py                       # default brand_id=1729
    uv run ingest_missing_videos.py --brand-id 1729
    uv run ingest_missing_videos.py --brand-id 1729 --dry-run
"""

import argparse
import asyncio
import logging
import os

from pymilvus import Collection, connections

from db.opensearch import scan_video_ids
from utils.ingest import ingest_videos

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MILVUS_URI       = os.environ.get("MILVUS_URI", "http://localhost:19530")
VIDEO_COLLECTION = os.environ.get("VIDEO_COLLECTION", "cohort_video_embeddings")
INGEST_BATCH     = 500


def get_milvus_video_ids(brand_id: int) -> set[str]:
    host, port = MILVUS_URI.replace("http://", "").split(":")
    connections.connect(host=host, port=port)

    collection = Collection(VIDEO_COLLECTION)
    collection.load()

    ids: set[str] = set()
    iterator = collection.query_iterator(
        expr=f"brand_id == {brand_id}",
        output_fields=["video_id"],
        batch_size=16384,
    )
    while True:
        batch = iterator.next()
        if not batch:
            iterator.close()
            break
        for r in batch:
            ids.add(r["video_id"])

    logger.info(f"Milvus has {len(ids)} videos for brand_id={brand_id}")
    return ids


async def main(brand_id: int, dry_run: bool) -> None:
    logger.info(f"Step 1/3 — Scanning OpenSearch for brand_id={brand_id}...")
    os_ids = await scan_video_ids(brand_id=brand_id)
    logger.info(f"  OpenSearch: {len(os_ids)} videos for brand_id={brand_id}")

    if not os_ids:
        logger.warning("No videos found in OpenSearch for this brand. Trying without brand filter...")
        os_ids = await scan_video_ids(brand_id=None)
        logger.info(f"  OpenSearch (all brands): {len(os_ids)} videos")

    logger.info("Step 2/3 — Fetching video IDs already in Milvus...")
    milvus_ids = get_milvus_video_ids(brand_id)

    missing = [vid for vid in os_ids if vid not in milvus_ids]
    logger.info(f"  Missing (OpenSearch - Milvus): {len(missing)} videos")

    if not missing:
        logger.info("Nothing to ingest — Milvus is up to date for this brand")
        return

    if dry_run:
        logger.info(f"DRY RUN — would ingest {len(missing)} videos (brand_id={brand_id}, has_engagement=False)")
        logger.info(f"  Sample IDs: {missing[:5]}")
        return

    logger.info(f"Step 3/3 — Ingesting {len(missing)} videos in batches of {INGEST_BATCH}...")
    total = 0
    n_batches = (len(missing) + INGEST_BATCH - 1) // INGEST_BATCH
    for i in range(0, len(missing), INGEST_BATCH):
        batch = missing[i : i + INGEST_BATCH]
        logger.info(f"  Batch {i // INGEST_BATCH + 1}/{n_batches}...")
        n = await ingest_videos(video_ids=batch, brand_id=brand_id, has_engagement=False)
        total += n

    logger.info(f"Done — ingested {total}/{len(missing)} videos into Milvus (has_engagement=False)")


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--brand-id", type=int, default=1729)
    parser.add_argument("--dry-run",  action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.brand_id, args.dry_run))


run()
