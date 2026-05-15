"""
Migrates video_embeddings collection to add has_engagement field.
Uses video_id as primary key so upserts work correctly.

Steps:
  1. Read all records from existing collection
  2. Create new collection with video_id as PK + has_engagement field
  3. Insert all records with has_engagement=True
  4. Rename: old → video_embeddings_backup, new → video_embeddings

Recovery (if a previous run left things broken):
  Run with --restore-backup to swap backup back as active collection.

Usage:
    uv run migrate_video_collection.py
    uv run migrate_video_collection.py --restore-backup
"""

import argparse
import logging
import os
import time

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MILVUS_URI     = os.environ.get("MILVUS_URI", "http://localhost:19530")
OLD_COLLECTION = os.environ.get("VIDEO_COLLECTION", "video_embeddings")
NEW_COLLECTION = f"{OLD_COLLECTION}_new"
BACKUP_NAME    = f"{OLD_COLLECTION}_backup"
EMBEDDING_DIM  = 512
BATCH          = 1000


def connect() -> None:
    host, port = MILVUS_URI.replace("http://", "").split(":")
    connections.connect(host=host, port=port)
    logger.info(f"Connected to Milvus at {MILVUS_URI}")


def fetch_all_records(collection_name: str) -> list[dict]:
    collection = Collection(collection_name)
    collection.load()

    records = []
    iterator = collection.query_iterator(
        expr="brand_id >= 0",
        output_fields=["video_id", "embedding", "brand_id", "updated_at"],
        batch_size=BATCH,
    )
    while True:
        batch = iterator.next()
        if not batch:
            iterator.close()
            break
        records.extend(batch)
        logger.info(f"  Fetched {len(records)} records...")

    logger.info(f"Total: {len(records)} records from '{collection_name}'")
    return records


def create_new_collection(name: str) -> Collection:
    fields = [
        # video_id as primary key — enables upsert-by-video_id
        FieldSchema(name="video_id",       dtype=DataType.VARCHAR,      max_length=256, is_primary=True, auto_id=False),
        FieldSchema(name="embedding",      dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema(name="brand_id",       dtype=DataType.INT64),
        FieldSchema(name="has_engagement", dtype=DataType.BOOL),
        FieldSchema(name="updated_at",     dtype=DataType.INT64),
    ]
    schema     = CollectionSchema(fields=fields, description="Video embeddings")
    collection = Collection(name=name, schema=schema)
    logger.info(f"Created collection '{name}' (PK: video_id)")
    return collection


def insert_records(collection: Collection, records: list[dict]) -> None:
    now = int(time.time())
    for i in range(0, len(records), BATCH):
        batch = records[i : i + BATCH]
        rows = [
            {
                "video_id":      r["video_id"],
                "embedding":     r["embedding"],
                "brand_id":      int(r["brand_id"]),
                "has_engagement": True,
                "updated_at":    int(r.get("updated_at", now)),
            }
            for r in batch
        ]
        collection.insert(rows)
        logger.info(f"  Inserted {min(i + BATCH, len(records))}/{len(records)}...")

    collection.flush()
    logger.info(f"Flushed {len(records)} records into '{collection.name}'")


def build_index(collection: Collection) -> None:
    collection.create_index(
        field_name="embedding",
        index_params={
            "metric_type": "IP",
            "index_type":  "IVF_FLAT",
            "params":      {"nlist": 128},
        },
    )
    logger.info(f"Index built on '{collection.name}'")


def restore_backup() -> None:
    if not utility.has_collection(BACKUP_NAME):
        logger.error(f"No backup '{BACKUP_NAME}' found — nothing to restore")
        return
    temp = f"{OLD_COLLECTION}_broken"
    if utility.has_collection(OLD_COLLECTION):
        utility.rename_collection(OLD_COLLECTION, temp)
        logger.info(f"  Moved broken '{OLD_COLLECTION}' → '{temp}'")
    utility.rename_collection(BACKUP_NAME, OLD_COLLECTION)
    logger.info(f"  Restored '{BACKUP_NAME}' → '{OLD_COLLECTION}'")
    if utility.has_collection(temp):
        utility.drop_collection(temp)
        logger.info(f"  Dropped '{temp}'")
    logger.info("Restore complete — original collection is active again")


def migrate() -> None:
    if utility.has_collection(BACKUP_NAME):
        logger.error(f"Backup '{BACKUP_NAME}' already exists — run with --restore-backup first, then re-run migration")
        return

    if utility.has_collection(NEW_COLLECTION):
        logger.warning(f"Dropping leftover '{NEW_COLLECTION}'...")
        utility.drop_collection(NEW_COLLECTION)

    logger.info(f"Step 1/4 — Reading records from '{OLD_COLLECTION}'...")
    records = fetch_all_records(OLD_COLLECTION)

    logger.info(f"Step 2/4 — Creating '{NEW_COLLECTION}' with video_id as PK...")
    new_col = create_new_collection(NEW_COLLECTION)

    logger.info(f"Step 3/4 — Inserting {len(records)} records (has_engagement=True)...")
    insert_records(new_col, records)
    build_index(new_col)

    logger.info("Step 4/4 — Renaming collections...")
    utility.rename_collection(OLD_COLLECTION, BACKUP_NAME)
    utility.rename_collection(NEW_COLLECTION, OLD_COLLECTION)
    logger.info(f"  Active : {OLD_COLLECTION}  (video_id PK, has_engagement field)")
    logger.info(f"  Backup : {BACKUP_NAME}      (safe to drop after verification)")
    logger.info("Migration complete.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore-backup", action="store_true", help="Restore original collection from backup")
    args = parser.parse_args()

    connect()
    if args.restore_backup:
        restore_backup()
    else:
        migrate()


main()
