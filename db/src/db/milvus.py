import logging
import os
import time

import numpy as np
from pymilvus import DataType, MilvusClient

logger = logging.getLogger(__name__)

_MILVUS_URI        = os.environ.get("MILVUS_URI", "http://localhost:19530")
_VIDEO_COLLECTION  = os.environ.get("VIDEO_COLLECTION", "cohort_video_embeddings")
_USER_COLLECTION   = os.environ.get("USER_COLLECTION", "cohort_user_embeddings")
_EMBEDDING_DIM     = 512
_BATCH             = 1000


def _client() -> MilvusClient:
    return MilvusClient(uri=_MILVUS_URI)


def _ensure_video_collection(client: MilvusClient, collection: str) -> None:
    if client.has_collection(collection):
        return
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("video_id",       DataType.VARCHAR,      is_primary=True, max_length=256)
    schema.add_field("embedding",      DataType.FLOAT_VECTOR, dim=_EMBEDDING_DIM)
    schema.add_field("brand_id",       DataType.INT64)
    schema.add_field("has_engagement", DataType.BOOL)
    schema.add_field("updated_at",     DataType.INT64)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index("embedding", metric_type="IP", index_type="IVF_FLAT", params={"nlist": 128})

    client.create_collection(collection, schema=schema, index_params=index_params)
    logger.info(f"Created Milvus collection: {collection}")


def _ensure_user_collection(client: MilvusClient, collection: str) -> None:
    if client.has_collection(collection):
        return
    schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("user_brand_key", DataType.VARCHAR,      is_primary=True, max_length=200)
    schema.add_field("embedding",      DataType.FLOAT_VECTOR, dim=_EMBEDDING_DIM)
    schema.add_field("user_id",        DataType.VARCHAR,       max_length=100)
    schema.add_field("brand_id",       DataType.INT64)
    schema.add_field("cohort_ids",     DataType.VARCHAR,       max_length=100)
    schema.add_field("updated_at",     DataType.INT64)

    index_params = MilvusClient.prepare_index_params()
    index_params.add_index("embedding", metric_type="IP", index_type="IVF_FLAT", params={"nlist": 128})

    client.create_collection(collection, schema=schema, index_params=index_params)
    logger.info(f"Created Milvus collection: {collection}")


def drop_collection(collection: str) -> None:
    """Drop a Milvus collection if it exists. Call before retraining to clear stale embeddings."""
    client = _client()
    if client.has_collection(collection):
        client.drop_collection(collection)
        logger.info(f"Dropped Milvus collection: {collection}")
    else:
        logger.info(f"Collection {collection} does not exist, nothing to drop")


def upsert_video_embeddings(
    finetuned_videos: dict[str, np.ndarray],
    video_brand_map: dict[str, int],
    engaged_video_ids: set[str] | None = None,
    collection: str | None = None,
) -> None:
    """
    engaged_video_ids: videos that appeared in training events → has_engagement=True.
    If None, all videos are marked as has_engagement=True (backwards compat).
    Videos not in the set are cold-start (content-only embedding) → has_engagement=False.
    """
    coll   = collection or _VIDEO_COLLECTION
    client = _client()
    _ensure_video_collection(client, coll)
    now    = int(time.time())
    records = [
        {
            "video_id":       str(vid_id),
            "embedding":      emb.tolist(),
            "brand_id":       int(video_brand_map.get(vid_id, 0)),
            "has_engagement": True if engaged_video_ids is None else str(vid_id) in engaged_video_ids,
            "updated_at":     now,
        }
        for vid_id, emb in finetuned_videos.items()
    ]
    for i in range(0, len(records), _BATCH):
        client.upsert(coll, records[i : i + _BATCH])
    n_cold = sum(1 for r in records if not r["has_engagement"])
    logger.info(f"Upserted {len(records)} video embeddings → Milvus:{coll} ({n_cold} cold-start)")


def upsert_user_embeddings(
    user_embeddings: dict[str, np.ndarray],   # key = "user_id::brand_id"
    user_cohort_map: dict[str, list[int]],    # key = "user_id::brand_id" → [cohort_ids]
    collection: str | None = None,
) -> None:
    coll   = collection or _USER_COLLECTION
    client = _client()
    _ensure_user_collection(client, coll)
    now    = int(time.time())
    records = []
    for key, emb in user_embeddings.items():
        user_id, brand_id = key.split("::", 1)
        cohorts = user_cohort_map.get(key, [0])
        records.append({
            "user_brand_key": key,
            "user_id":        user_id,
            "brand_id":       int(brand_id),
            "embedding":      emb.tolist(),
            "cohort_ids":     ",".join(str(c) for c in cohorts),
            "updated_at":     now,
        })
    for i in range(0, len(records), _BATCH):
        client.upsert(coll, records[i : i + _BATCH])
    logger.info(f"Upserted {len(records)} user embeddings → Milvus:{coll}")
