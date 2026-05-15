import asyncio
import logging
import os
from pathlib import Path

import numpy as np
import torch

from db.milvus import upsert_video_embeddings
from db.opensearch import get_videos_by_ids
from models.two_tower import VideoTower
from utils.embeddings import embed_videos

logger = logging.getLogger(__name__)

_MODEL_PATH   = Path(os.environ.get("MODEL_PATH", "cache/two_tower.pt"))
_BACKBONE_DIM = 768
_OUTPUT_DIM   = 512

_video_tower: VideoTower | None = None


def _load_video_tower() -> VideoTower:
    global _video_tower
    if _video_tower is not None:
        return _video_tower

    if not _MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {_MODEL_PATH}")

    state_dict = torch.load(_MODEL_PATH, map_location="cpu")
    tower_state = {
        k.replace("video_tower.", ""): v
        for k, v in state_dict.items()
        if k.startswith("video_tower.")
    }

    tower = VideoTower(backbone_dim=_BACKBONE_DIM, output_dim=_OUTPUT_DIM)
    tower.load_state_dict(tower_state)
    tower.eval()
    logger.info(f"Loaded VideoTower from {_MODEL_PATH}")

    _video_tower = tower
    return _video_tower


def _project_embeddings(
    raw_embeddings: dict[str, np.ndarray],
    tower: VideoTower,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for vid_id, emb in raw_embeddings.items():
            t = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
            result[vid_id] = tower(t).squeeze(0).cpu().numpy()
    return result


async def ingest_videos(
    video_ids: list[str],
    brand_id: int,
    has_engagement: bool,
) -> int:
    """
    Fetches video content from OpenSearch, computes 512d embeddings via
    backbone + VideoTower, and upserts into Milvus.

    Returns the number of videos successfully ingested.
    """
    if not video_ids:
        return 0

    logger.info(f"Ingesting {len(video_ids)} videos (brand_id={brand_id}, has_engagement={has_engagement})...")

    # Step 1: fetch content from OpenSearch
    videos = await get_videos_by_ids(video_ids)
    if not videos:
        logger.warning("No video content found in OpenSearch for the given IDs")
        return 0

    # Step 2: backbone embedding (768d weighted combination of text fields)
    raw_embeddings = embed_videos(videos)

    # Step 3: project through VideoTower MLP → 512d
    tower = _load_video_tower()
    finetuned = _project_embeddings(raw_embeddings, tower)

    # Step 4: upsert to Milvus
    video_brand_map = {vid_id: brand_id for vid_id in finetuned}
    engaged_ids     = set(finetuned.keys()) if has_engagement else set()
    upsert_video_embeddings(finetuned, video_brand_map, engaged_video_ids=engaged_ids)

    logger.info(f"Ingested {len(finetuned)} videos successfully")
    return len(finetuned)


def ingest_videos_sync(
    video_ids: list[str],
    brand_id: int,
    has_engagement: bool,
) -> int:
    """Sync wrapper around ingest_videos for non-async call sites."""
    return asyncio.run(ingest_videos(video_ids, brand_id, has_engagement))
