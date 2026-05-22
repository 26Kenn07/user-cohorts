"""
Page context utilities: extract text from URL metadata, embed it,
and compute per-(user, brand) average context embeddings.
"""
import logging
import re

import numpy as np
import pandas as pd

from .embeddings import _get_model

logger = logging.getLogger(__name__)


def extract_page_context(metadata: dict) -> str:
    """
    Returns a comma-separated keyword string from URL metadata.

    Priority order:
      1. Pre-extracted 'bert_keywords' field (set by enrich_metadata_keywords.py)
      2. Raw text fallback: title + description + explicit keyword/tag fields
    """
    # Fast path: pre-extracted KeyBERT keywords already stored in the entry
    bert_kws = metadata.get("bert_keywords")
    if bert_kws is not None:
        tokens = [str(k).strip().lower() for k in bert_kws if k]
        if tokens:
            return ", ".join(tokens)

    m = metadata.get("metadata", metadata) or {}
    parts: list[str] = []

    if title := m.get("title"):
        clean = re.split(r"\s[|\-–]\s", title)[0].strip()
        if clean:
            parts.append(clean)

    if desc := m.get("description"):
        parts.append(desc.strip())

    for field in ("keywords", "categories", "tags"):
        vals = m.get(field)
        if vals and isinstance(vals, list):
            parts.extend(str(v).strip() for v in vals if v)

    if section := m.get("section"):
        parts.append(str(section).strip())

    seen: set[str] = set()
    tokens: list[str] = []
    for part in ", ".join(parts).split(", "):
        t = part.strip().lower()
        if t and t not in seen:
            seen.add(t)
            tokens.append(t)

    return ", ".join(tokens)


def extract_keywords_keybert(text: str, top_n: int = 10) -> list[str]:
    """
    Extracts discriminative keyphrases from raw text using KeyBERT
    backed by the shared all-mpnet-base-v2 backbone.
    Used for live extraction on URLs not present in combined_metadata.json.
    """
    from keybert import KeyBERT
    kw_model = KeyBERT(model=_get_model())
    kws = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=top_n,
        use_mmr=True,
        diversity=0.5,
    )
    return [kw for kw, _ in kws]


def embed_page_contexts(
    url_context_map: dict[str, str],
    batch_size: int = 256,
) -> dict[str, np.ndarray]:
    """
    Embeds page context strings using the shared ST backbone (all-mpnet-base-v2).
    Returns url → 768d normalized embedding.
    """
    model = _get_model()
    urls  = list(url_context_map.keys())
    texts = [url_context_map[u] or "unknown page" for u in urls]

    logger.info(f"Embedding {len(urls)} unique page URLs...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return {url: embeddings[i].astype(np.float32) for i, url in enumerate(urls)}


def get_user_page_ctx_embs(
    events_df: pd.DataFrame,
    url_emb_map: dict[str, np.ndarray],
    emb_dim: int = 768,
) -> dict[str, np.ndarray]:
    """
    Averages page context embeddings across all URLs seen by each (user, brand).
    Returns key "user_id::brand_id" → 768d float32 array.
    Users with no matched URLs get a zero vector.
    """
    result: dict[str, np.ndarray] = {}
    for (user_id, brand_id), group in events_df.groupby(["user_id", "brand_id"]):
        embs = [
            url_emb_map[url]
            for url in group["url"].dropna().astype(str).unique()
            if url in url_emb_map
        ]
        key = f"{user_id}::{brand_id}"
        result[key] = np.mean(embs, axis=0).astype(np.float32) if embs else np.zeros(emb_dim, dtype=np.float32)

    covered = sum(1 for v in result.values() if np.any(v != 0))
    logger.info(f"Page context: {covered}/{len(result)} (user, brand) pairs have ≥1 URL matched")
    return result
