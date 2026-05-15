"""
(Re)creates the user_embeddings Milvus collection with the multi-brand schema.

Schema:
  user_brand_key  VARCHAR(512)  — PK, format "user_id::brand_id"
  user_id         VARCHAR(256)
  brand_id        INT64
  embedding       FLOAT_VECTOR(512)
  cohort_id       INT64
  updated_at      INT64

Run this once before retraining to apply the multi-brand schema change.
WARNING: drops the existing user_embeddings collection if it exists.

Usage:
    uv run setup_milvus.py
"""

import logging
import os

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

MILVUS_URI       = os.environ.get("MILVUS_URI", "http://localhost:19530")
USER_COLLECTION  = os.environ.get("USER_COLLECTION", "user_embeddings")
EMBEDDING_DIM    = 512


def connect() -> None:
    host, port = MILVUS_URI.replace("http://", "").split(":")
    connections.connect(host=host, port=port)
    logger.info(f"Connected to Milvus at {MILVUS_URI}")


def recreate_user_collection() -> None:
    if utility.has_collection(USER_COLLECTION):
        utility.drop_collection(USER_COLLECTION)
        logger.info(f"Dropped existing '{USER_COLLECTION}'")

    fields = [
        FieldSchema(name="user_brand_key", dtype=DataType.VARCHAR,      max_length=512,  is_primary=True, auto_id=False),
        FieldSchema(name="user_id",        dtype=DataType.VARCHAR,      max_length=256),
        FieldSchema(name="brand_id",       dtype=DataType.INT64),
        FieldSchema(name="embedding",      dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        FieldSchema(name="cohort_ids",     dtype=DataType.VARCHAR,      max_length=64),   # e.g. "0" or "0,2"
        FieldSchema(name="updated_at",     dtype=DataType.INT64),
    ]
    schema     = CollectionSchema(fields=fields, description="User embeddings (multi-brand, multi-cohort)")
    collection = Collection(name=USER_COLLECTION, schema=schema)
    logger.info(f"Created '{USER_COLLECTION}' (PK: user_brand_key)")

    collection.create_index(
        field_name="embedding",
        index_params={
            "metric_type": "IP",
            "index_type":  "IVF_FLAT",
            "params":      {"nlist": 128},
        },
    )
    logger.info(f"Index built on '{USER_COLLECTION}.embedding'")
    logger.info("Done — run 'uv run test.py' to populate with fresh embeddings")


def main() -> None:
    connect()
    recreate_user_collection()


main()
