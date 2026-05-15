import logging
import os
import time

import numpy as np
from pymilvus import MilvusClient

logger = logging.getLogger(__name__)

_MILVUS_URI        = os.environ.get("MILVUS_URI", "http://localhost:19530")
_VIDEO_COLLECTION  = os.environ.get("VIDEO_COLLECTION", "video_embeddings")
_USER_COLLECTION   = os.environ.get("USER_COLLECTION", "user_embeddings")
_BATCH             = 1000


def _client() -> MilvusClient:
    return MilvusClient(uri=_MILVUS_URI)


def upsert_video_embeddings(
    finetuned_videos: dict[str, np.ndarray],
    video_brand_map: dict[str, int],
    engaged_video_ids: set[str] | None = None,
) -> None:
    """
    engaged_video_ids: videos that appeared in training events → has_engagement=True.
    If None, all videos are marked as has_engagement=True (backwards compat).
    Videos not in the set are cold-start (content-only embedding) → has_engagement=False.
    """
    client = _client()
    now    = int(time.time())
    records = [
        {
            "video_id":      str(vid_id),
            "embedding":     emb.tolist(),
            "brand_id":      int(video_brand_map.get(vid_id, 0)),
            "has_engagement": True if engaged_video_ids is None else str(vid_id) in engaged_video_ids,
            "updated_at":    now,
        }
        for vid_id, emb in finetuned_videos.items()
    ]
    for i in range(0, len(records), _BATCH):
        client.upsert(_VIDEO_COLLECTION, records[i : i + _BATCH])
    n_cold = sum(1 for r in records if not r["has_engagement"])
    logger.info(f"Upserted {len(records)} video embeddings → Milvus:{_VIDEO_COLLECTION} ({n_cold} cold-start)")


def upsert_user_embeddings(
    user_embeddings: dict[str, np.ndarray],   # key = "user_id::brand_id"
    user_cohort_map: dict[str, list[int]],    # key = "user_id::brand_id" → [cohort_ids]
) -> None:
    client = _client()
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
        client.upsert(_USER_COLLECTION, records[i : i + _BATCH])
    logger.info(f"Upserted {len(records)} user embeddings → Milvus:{_USER_COLLECTION}")
