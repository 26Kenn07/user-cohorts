import logging
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None

WEIGHTS = {
    "transcript": 0.4,
    "description_text": 0.3,
    "video_gen_description": 0.2,
    "keywords": 0.1,
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
        logger.info(f"Loading sentence-transformer model on {device}...")
        _model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    return _model


def _embed_texts(texts: list[str]) -> np.ndarray:
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)


def _embed_video(video: dict[str, Any]) -> np.ndarray:
    model = _get_model()
    dim = model.get_embedding_dimension()
    assert dim is not None

    weighted_sum = np.zeros(dim)
    total_weight = 0.0

    # Embed transcript, description, ai description — each as one text block
    for field, weight in [
        ("transcript", WEIGHTS["transcript"]),
        ("description_text", WEIGHTS["description_text"]),
        ("video_gen_description", WEIGHTS["video_gen_description"]),
    ]:
        text = video.get(field, "").strip()
        if not text:
            continue
        embedding = _embed_texts([text])[0]
        weighted_sum += embedding * weight
        total_weight += weight

    # Embed each keyword separately, then average them
    keywords: list[str] = [kw for kw in (video.get("keywords") or []) if kw.strip()]
    if keywords:
        keyword_embeddings = _embed_texts(keywords)
        keyword_avg = keyword_embeddings.mean(axis=0)
        weighted_sum += keyword_avg * WEIGHTS["keywords"]
        total_weight += WEIGHTS["keywords"]

    if total_weight == 0:
        return np.zeros(dim)

    final = weighted_sum / total_weight
    # L2 normalize so dot product == cosine similarity in Milvus
    norm = np.linalg.norm(final)
    return final / norm if norm > 0 else final


def embed_videos(videos: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    """
    Returns a dict of video_id -> embedding (384d, L2 normalized).
    Videos with no content get a zero vector and are flagged.
    """
    logger.info(f"Embedding {len(videos)} videos...")
    result = {}
    zero_count = 0

    for video in videos:
        vid_id = video["video_id"]
        embedding = _embed_video(video)
        result[vid_id] = embedding
        if np.all(embedding == 0):
            zero_count += 1
            logger.warning(f"video_id {vid_id} has no content — zero vector assigned")

    if zero_count:
        logger.warning(f"{zero_count}/{len(videos)} videos have zero embeddings")

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
