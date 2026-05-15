import logging
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
from tqdm import tqdm

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None

BACKBONE_MODEL = "all-mpnet-base-v2"   # 768d, 110M params — significantly better than MiniLM

WEIGHTS = {
    "transcript": 0.4,
    "description_text": 0.3,
    "video_gen_description": 0.25,
    "keywords": 0.05,
}


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        import torch
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        logger.info(f"Loading {BACKBONE_MODEL} on {device}...")
        _model = SentenceTransformer(BACKBONE_MODEL, device=device)
    return _model


_TEXT_FIELDS = [
    ("transcript", WEIGHTS["transcript"]),
    ("description_text", WEIGHTS["description_text"]),
    ("video_gen_description", WEIGHTS["video_gen_description"]),
]


def embed_videos(
    videos: list[dict[str, Any]],
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    """
    Returns a dict of video_id -> embedding (768d, L2 normalized).
    Batches all text into a single GPU encode pass per field for speed.
    """
    model = _get_model()
    dim = model.get_embedding_dimension()
    assert dim is not None

    n = len(videos)
    logger.info(f"Embedding {n} videos (batched, batch_size={batch_size})...")

    field_embeddings: dict[str, np.ndarray] = {}
    field_masks: dict[str, np.ndarray] = {}

    for field, _ in _TEXT_FIELDS:
        texts: list[str] = []
        mask = np.zeros(n, dtype=bool)
        for i, v in enumerate(videos):
            t = v.get(field, "").strip()
            texts.append(t if t else "")
            if t:
                mask[i] = True

        logger.info(f"  Encoding field '{field}' ({int(mask.sum())} non-empty)...")
        embs = model.encode(
            texts, batch_size=batch_size, convert_to_numpy=True,
            show_progress_bar=True,
        )
        field_embeddings[field] = embs
        field_masks[field] = mask

    kw_texts: list[str] = []
    kw_indices: list[int] = []
    kw_counts: list[int] = []
    for i, v in enumerate(videos):
        keywords = [kw for kw in (v.get("keywords") or []) if kw.strip()]
        if keywords:
            kw_indices.append(i)
            kw_counts.append(len(keywords))
            kw_texts.extend(keywords)

    kw_avg = np.zeros((n, dim), dtype=np.float32)
    kw_mask = np.zeros(n, dtype=bool)
    if kw_texts:
        logger.info(f"  Encoding {len(kw_texts)} keywords for {len(kw_indices)} videos...")
        all_kw_embs = model.encode(
            kw_texts, batch_size=batch_size, convert_to_numpy=True,
            show_progress_bar=True,
        )
        offset = 0
        for idx, count in zip(kw_indices, kw_counts):
            kw_avg[idx] = all_kw_embs[offset : offset + count].mean(axis=0)
            kw_mask[idx] = True
            offset += count

    logger.info("  Combining weighted embeddings...")
    result: dict[str, np.ndarray] = {}
    zero_count = 0

    for i, v in enumerate(tqdm(videos, desc="Combining embeddings")):
        weighted_sum = np.zeros(dim, dtype=np.float32)
        total_weight = 0.0

        for field, weight in _TEXT_FIELDS:
            if field_masks[field][i]:
                weighted_sum += field_embeddings[field][i] * weight
                total_weight += weight

        if kw_mask[i]:
            weighted_sum += kw_avg[i] * WEIGHTS["keywords"]
            total_weight += WEIGHTS["keywords"]

        if total_weight == 0:
            result[v["video_id"]] = np.zeros(dim)
            zero_count += 1
            continue

        final = weighted_sum / total_weight
        norm = np.linalg.norm(final)
        result[v["video_id"]] = final / norm if norm > 0 else final

    if zero_count:
        logger.warning(f"{zero_count}/{n} videos have zero embeddings")

    logger.info("Video embedding complete.")
    return result


def embed_users(
    engagement_df: pd.DataFrame,
    video_embeddings: dict[str, np.ndarray],
    dim: int = 384,
) -> dict[str, np.ndarray]:
    """
    Computes user embeddings as engagement-score-weighted average of video embeddings.
    Groups by user_id — users with no valid video embeddings are skipped.
    """
    user_embeddings: dict[str, np.ndarray] = {}

    for user_id, group in engagement_df.groupby("user_id"):
        weighted_sum = np.zeros(dim)
        total_weight = 0.0

        for _, row in group.iterrows():
            vid_emb = video_embeddings.get(row["video_id"])
            if vid_emb is None or np.all(vid_emb == 0):
                continue
            weighted_sum += vid_emb * row["score"]
            total_weight += row["score"]

        if total_weight == 0:
            continue

        user_emb = weighted_sum / total_weight
        norm = np.linalg.norm(user_emb)
        user_embeddings[str(user_id)] = user_emb / norm if norm > 0 else user_emb

    logger.info(f"Computed embeddings for {len(user_embeddings)} users")
    return user_embeddings
