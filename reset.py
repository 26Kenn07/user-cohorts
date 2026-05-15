"""
Resets all training state before a fresh run of test.py.

Clears:
  - cache/*.pkl and cache/*.pt  (all cached data, embeddings, model checkpoints)
  - Milvus video_embeddings and user_embeddings collections

Preserves:
  - new_ck_user_events.csv  (raw event data)
  - new_video_data.csv      (raw video metadata)

Usage:
    uv run reset.py              # clear everything
    uv run reset.py --keep-data  # clear model/embeddings but keep CSV caches
                                 # (all_events, train_df, test_df, videos, video_embeddings)
"""

import argparse
import logging
import shutil
from pathlib import Path

from pymilvus import MilvusClient
from utils.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = Path("cache")

# Caches that are expensive to rebuild (CSV parsing + sentence-transformer embedding)
DATA_CACHES = {
    "all_events.pkl",
    "train_df.pkl",
    "test_df.pkl",
    "videos.pkl",
    "video_embeddings.pkl",
}

# Caches that are cheap to rebuild (training outputs)
MODEL_CACHES = {
    "two_tower.pt",
    "cohort_profiles.pkl",
    "video_cohort_map.pkl",
    "user_cohort_map.pkl",
    "user_embeddings_all.pkl",
}


def clear_cache(keep_data: bool) -> None:
    if not CACHE_DIR.exists():
        logger.info("cache/ directory does not exist — nothing to clear")
        return

    removed = []
    for path in sorted(CACHE_DIR.iterdir()):
        if keep_data and path.name in DATA_CACHES:
            logger.info(f"  keeping  {path.name}")
            continue
        path.unlink()
        removed.append(path.name)

    if removed:
        logger.info(f"Removed {len(removed)} cache file(s): {', '.join(removed)}")
    else:
        logger.info("No cache files removed")


def clear_milvus() -> None:
    uri = settings.milvus.uri
    video_col = settings.milvus.video_collection
    user_col  = settings.milvus.user_collection

    logger.info(f"Connecting to Milvus at {uri}...")
    client = MilvusClient(uri=uri)

    for col in [video_col, user_col]:
        try:
            stats = client.get_collection_stats(col)
            row_count = stats.get("row_count", "?")
            client.load_collection(col)
            client.delete(collection_name=col, filter="updated_at >= 0")
            logger.info(f"  Cleared {col} ({row_count} rows deleted)")
        except Exception as e:
            logger.warning(f"  Could not clear {col}: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reset training state for test.py")
    parser.add_argument(
        "--keep-data",
        action="store_true",
        help="Keep expensive data caches (all_events, train/test splits, video embeddings)",
    )
    args = parser.parse_args()

    logger.info("=== Resetting training state ===")
    logger.info(f"Mode: {'keep data caches' if args.keep_data else 'full reset'}")

    clear_cache(keep_data=args.keep_data)
    clear_milvus()

    logger.info("Done. Run `uv run test.py` to retrain.")


main()
