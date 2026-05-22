"""
Inference against the web-only model variants, with optional real-time page context.

Modes (same as infer_v2.py):
  user-to-video  — recommended videos for a user
  video-to-video — similar videos to a source video
  video-to-user  — users most likely to engage with a video
  user-to-user   — users with similar taste
  list-cohorts   — print all cohort labels and descriptions

Flags:
  --ctx            Use the page-context model (cache_web_ctx / *_web_ctx Milvus collections).
                   Default: no-context model (cache_web_no_ctx / *_web_no_ctx collections).

  --url <url>      Inject this page's context into the user embedding at query time.
                   Only applies to user-to-video mode + requires --ctx.
                   Looks up the URL in combined_metadata.json, prints the extracted
                   context string, then shows two ranked lists side-by-side:
                     [A] Stored embedding (historical page-context average)
                     [B] Recomputed embedding with the supplied URL context
                   so you can see exactly what the page changes.

Examples:
    uv run infer_web.py user-to-video  <user_id> --brand-id 2314
    uv run infer_web.py user-to-video  <user_id> --brand-id 2314 --ctx
    uv run infer_web.py user-to-video  <user_id> --brand-id 2314 --ctx --url "https://www.carlist.my/..."
    uv run infer_web.py video-to-video <video_id> --ctx
    uv run infer_web.py video-to-user  <video_id> --ctx
    uv run infer_web.py user-to-user   <user_id> --brand-id 2314 --ctx
    uv run infer_web.py list-cohorts   --ctx
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from pymilvus import MilvusClient

load_dotenv()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

LOG_FILE = Path("web_infer.log")


class _Tee:
    """Writes to both the original stdout and a log file simultaneously."""
    def __init__(self, stream, filepath: Path):
        self._stream = stream
        self._file = open(filepath, "a")

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def __getattr__(self, attr):
        return getattr(self._stream, attr)


sys.stdout = _Tee(sys.stdout, LOG_FILE)

# ---------------------------------------------------------------------------
# Constants — switched by --ctx flag
# ---------------------------------------------------------------------------

_CTX_CACHE       = Path("cache_web_ctx")
_CTX_VIDEO_COLL  = "video_embeddings_web_ctx"
_CTX_USER_COLL   = "user_embeddings_web_ctx"
_CTX_MODEL_FILE  = _CTX_CACHE / "two_tower_web_ctx.pt"

_NOCTX_CACHE      = Path("cache_web_no_ctx")
_NOCTX_VIDEO_COLL = "video_embeddings_web_no_ctx"
_NOCTX_USER_COLL  = "user_embeddings_web_no_ctx"

_RERANK_CACHE      = Path("cache_web_rerank")
_RERANK_VIDEO_COLL = "video_embeddings_web_rerank"
_RERANK_USER_COLL  = "user_embeddings_web_rerank"
_RERANK_LAMBDA     = 0.7

METADATA_FILE = Path("combined_metadata.json")
MILVUS_URI    = __import__("os").environ.get("MILVUS_URI", "http://localhost:19530")
DESC_WIDTH    = 200
W             = 72


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

_cache: dict = {}

def _load(name: str, cache_dir: Path):
    key = f"{cache_dir}/{name}"
    if key not in _cache:
        path = cache_dir / f"{name}.pkl"
        _cache[key] = pickle.load(open(path, "rb")) if path.exists() else None
    return _cache[key]


def _video_lookup(cache_dir: Path) -> dict[str, dict]:
    videos = _load("videos", cache_dir) or []
    return {v["video_id"]: v for v in videos}


def _cohort_lookup(cache_dir: Path) -> dict[int, dict]:
    profiles = _load("cohort_profiles", cache_dir) or []
    return {p["cohort_id"]: p for p in profiles}


# ---------------------------------------------------------------------------
# Milvus helpers
# ---------------------------------------------------------------------------

def _get_source(client: MilvusClient, collection: str, id_field: str, id_value: str, user_coll: str) -> dict:
    extra = ["user_id", "brand_id", "cohort_ids"] if collection == user_coll else ["brand_id", "has_engagement"]
    rows = client.query(
        collection_name=collection,
        filter=f'{id_field} == "{id_value}"',
        output_fields=["embedding"] + extra,
    )
    if not rows:
        print(f"Error: '{id_value}' not found in '{collection}'", file=sys.stderr)
        sys.exit(1)
    return rows[0]


def _try_get_source(client: MilvusClient, collection: str, id_field: str, id_value: str) -> dict | None:
    """Non-fatal version of _get_source — returns None if collection missing or user not found."""
    try:
        if not client.has_collection(collection):
            return None
        extra = ["user_id", "brand_id", "cohort_ids"]
        rows = client.query(
            collection_name=collection,
            filter=f'{id_field} == "{id_value}"',
            output_fields=["embedding"] + extra,
        )
        return rows[0] if rows else None
    except Exception:
        return None


def _search(
    client: MilvusClient,
    collection: str,
    embedding: list,
    id_field: str,
    extra_fields: list[str],
    top_k: int,
    brand_id: int | None = None,
    exclude_id: str | None = None,
) -> list[dict]:
    filter_expr = f"brand_id == {brand_id}" if brand_id is not None else ""
    limit   = top_k + 1 if exclude_id else top_k
    results = client.search(
        collection_name=collection,
        data=[embedding],
        anns_field="embedding",
        limit=limit,
        output_fields=[id_field] + extra_fields,
        filter=filter_expr,
        search_params={"metric_type": "IP", "params": {"nprobe": 128}},
    )[0]
    if exclude_id:
        results = [h for h in results if h["entity"][id_field] != exclude_id]
    return results[:top_k]


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _video_snippet(video_id: str, cache_dir: Path, width: int = DESC_WIDTH) -> str:
    meta = _video_lookup(cache_dir).get(video_id, {})
    def _clean(v) -> str:
        s = str(v).strip() if v else ""
        return "" if s.lower() in ("nan", "none", "") else s
    text = (
        _clean(meta.get("video_gen_description")) or
        _clean(meta.get("description_text")) or
        _clean(meta.get("transcript")) or ""
    ).replace("\n", " ")
    return (text[:width] + "…" if len(text) > width else text) if text else "(no description)"


def _cohort_label(cohort_ids_str: str, cache_dir: Path) -> str:
    by_id  = _cohort_lookup(cache_dir)
    ids    = [int(x) for x in cohort_ids_str.split(",") if x.strip().isdigit()]
    labels = [by_id[cid]["label"] for cid in ids if cid in by_id]
    return ", ".join(labels) if labels else cohort_ids_str


def _user_persona(user_id: str, brand_id: int, cohort_ids_str: str, cache_dir: Path) -> str:
    train_df = _load("train_df", cache_dir)
    if train_df is None:
        return "  (engagement data not available)"
    g = train_df[
        (train_df["user_id"].astype(str) == str(user_id)) &
        (train_df["brand_id"].astype(str) == str(brand_id))
    ]
    if g.empty:
        return "  (user not found in training data)"

    avg_watch      = g["watch_percentage"].mean()
    total_likes    = int(g["likes"].sum())
    total_shares   = int(g["shares"].sum())
    total_comments = int(g["comments"].sum())

    if total_shares > 0:
        style = f"active sharer ({total_shares} shares)"
    elif total_likes > 0:
        style = f"liker ({total_likes} likes)"
    elif avg_watch >= 70:
        style = "deep watcher (no explicit reactions)"
    else:
        style = "passive browser"

    g = g.copy()
    g["_score"] = (
        (g["watch_percentage"] / 100) * 1.0
        + g["views"].clip(upper=3) * 0.5
        + g["likes"] * 3.0
        + g["shares"] * 5.0
        + g["comments"] * 3.0
    )
    top_vids = g.nlargest(3, "_score")["video_id"].tolist()
    top_snippets = [f"    · {_video_snippet(v, cache_dir, 70)}" for v in top_vids]

    return "\n".join([
        f"  Videos watched  : {len(g)}",
        f"  Avg watch time  : {avg_watch:.1f}%",
        f"  Reactions       : {total_likes} likes · {total_shares} shares · {total_comments} comments",
        f"  Engagement style: {style}",
        f"  Content cohorts : {_cohort_label(cohort_ids_str, cache_dir)}",
        f"  Top engaged videos:",
    ] + (top_snippets or ["    (none)"]))


# ---------------------------------------------------------------------------
# Page-context helpers (--url support)
# ---------------------------------------------------------------------------

def _extract_url_context(url: str) -> tuple[str, str]:
    """
    Returns (context_string, source_label).
    Priority:
      1. bert_keywords stored in combined_metadata.json (set by enrich_metadata_keywords.py)
      2. Raw fallback fields (title/description/keywords) from combined_metadata.json
      3. Live-fetch via utils.metadata_extracter + KeyBERT extraction
    """
    if METADATA_FILE.exists():
        with open(METADATA_FILE) as f:
            raw: dict = json.load(f)

        entry = raw.get(url)
        if entry:
            # Fast path: pre-extracted KeyBERT keywords
            bert_kws = entry.get("bert_keywords")
            if bert_kws:
                ctx = ", ".join(str(k).strip().lower() for k in bert_kws if k)
                return ctx, "bert_keywords"

            # Fallback: build from raw metadata fields
            import re
            m = entry.get("metadata") or {}
            parts: list[str] = []
            found: list[str] = []
            if title := m.get("title"):
                clean = re.split(r"\s[|\-–]\s", title)[0].strip()
                if clean:
                    parts.append(clean)
                    found.append("title")
            if desc := m.get("description"):
                parts.append(desc.strip())
                found.append("description")
            for field in ("keywords", "categories", "tags"):
                vals = m.get(field)
                if vals and isinstance(vals, list):
                    parts.extend(str(v).strip() for v in vals if v)
                    found.append(field)
            if section := m.get("section"):
                parts.append(str(section).strip())
                found.append("section")
            seen: set[str] = set()
            tokens: list[str] = []
            for part in ", ".join(parts).split(", "):
                t = part.strip().lower()
                if t and t not in seen:
                    seen.add(t)
                    tokens.append(t)
            if tokens:
                return ", ".join(tokens), "+".join(found) if found else "no fields"

    # Live-fetch + KeyBERT extraction for URLs not in combined_metadata.json
    try:
        from utils.metadata_extracter import get_url_preview
        from utils.page_context import extract_keywords_keybert
        fetched = get_url_preview(url)
        if fetched:
            m2 = fetched.get("metadata", fetched) or {}
            parts2 = []
            if title2 := m2.get("title"):
                parts2.append(title2)
            if desc2 := m2.get("description"):
                parts2.append(desc2)
            for field2 in ("keywords", "categories", "tags"):
                vals2 = m2.get(field2)
                if vals2 and isinstance(vals2, list):
                    parts2.extend(str(v) for v in vals2 if v)
            source_text = " ".join(parts2)
            if source_text.strip():
                kws = extract_keywords_keybert(source_text)
                ctx = ", ".join(kws)
                return ctx, "live-fetch+keybert"
    except Exception:
        pass

    return "", "URL not in metadata"


def _embed_text(text: str) -> np.ndarray:
    from utils.embeddings import _get_model
    model = _get_model()
    emb = model.encode([text or "unknown page"], normalize_embeddings=True, convert_to_numpy=True)
    return emb[0].astype(np.float32)


def _recompute_user_embedding(
    user_id: str,
    brand_id: int,
    page_ctx_emb: np.ndarray,
    cache_dir: Path,
) -> np.ndarray:
    """Load the ctx model and recompute user embedding with a specific page context."""
    from models.two_tower import TwoTowerModel, IndexMaps, get_user_embedding
    from utils.engagement import compute_engagement_scores

    train_df = _load("train_df", cache_dir)
    if train_df is None:
        print("Error: train_df not in cache — run test_web_ctx.py first.", file=sys.stderr)
        sys.exit(1)

    index_maps    = IndexMaps(train_df)
    engagement_df = compute_engagement_scores(train_df)

    model = TwoTowerModel(
        n_users=index_maps.n_users,
        n_brands=index_maps.n_brands,
        backbone_dim=768,
        output_dim=512,
        temperature=10.0,
        use_page_ctx=True,
    )
    state = torch.load(_CTX_MODEL_FILE, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    user_rows = engagement_df[
        (engagement_df["user_id"].astype(str) == str(user_id)) &
        (engagement_df["brand_id"].astype(str) == str(brand_id))
    ]
    if user_rows.empty:
        avg_eng = np.array([0.5, 0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    else:
        avg_eng = np.array([
            float(user_rows["watch_percentage"].mean()) / 100.0,
            min(float(user_rows["views"].mean()), 5.0) / 5.0,
            float(user_rows["likes"].mean()),
            float(user_rows["shares"].mean()),
            float(user_rows["comments"].mean()),
            float(user_rows["link_clicks"].mean()) if "link_clicks" in user_rows.columns else 0.0,
        ], dtype=np.float32)

    user_idx  = index_maps.get_user_idx(str(user_id), str(brand_id))
    brand_idx = index_maps.get_brand_idx(str(brand_id))
    return get_user_embedding(model, user_idx, brand_idx, avg_eng, page_ctx_emb)


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------

def run_user_to_video(
    client: MilvusClient,
    user_id: str,
    brand_id: int,
    top_k: int,
    video_coll: str,
    user_coll: str,
    cache_dir: Path,
    url: str | None,
    keywords: str | None = None,
) -> None:
    user_brand_key = f"{user_id}::{brand_id}"
    source = _get_source(client, user_coll, "user_brand_key", user_brand_key, user_coll)
    persona = _user_persona(user_id, brand_id, source.get("cohort_ids", "0"), cache_dir)
    print('-'*150)
    print(f"\n{'=' * W}")
    print(f"  USER PERSONA")
    print(f"{'─' * W}")
    print(f"  User  : {user_id}")
    print(f"  Brand : {brand_id}")
    print(persona)

    ctx_text: str | None = None
    ctx_source: str = ""
    if keywords:
        ctx_text   = keywords.strip()
        ctx_source = "direct-keywords"
    elif url:
        ctx_text, ctx_source = _extract_url_context(url)

    if ctx_text is not None:
        print(f"\n{'─' * W}")
        label = f"URL      : {url}" if url else "Keywords : (direct)"
        print(f"  PAGE CONTEXT  ({ctx_source})")
        print(f"  {label}")
        print(f"  Text : {ctx_text[:200]}{'…' if len(ctx_text) > 200 else ''}")

        page_ctx_emb  = _embed_text(ctx_text)
        url_embedding = _recompute_user_embedding(user_id, brand_id, page_ctx_emb, cache_dir)

    print(f"\n{'=' * W}")

    if ctx_text is not None:
        # Side-by-side: stored (historical avg) vs URL-recomputed
        stored_hits = _search(
            client, video_coll, source["embedding"],
            "video_id", ["brand_id"], top_k, brand_id,
        )
        url_hits = _search(
            client, video_coll, url_embedding.tolist(),
            "video_id", ["brand_id"], top_k, brand_id,
        )

        stored_ids = [h["entity"]["video_id"] for h in stored_hits]
        url_ids    = [h["entity"]["video_id"] for h in url_hits]

        print(f"  [A] TOP {top_k} — stored embedding (historical page avg)")
        print(f"{'─' * W}")
        for i, hit in enumerate(stored_hits, 1):
            vid_id  = hit["entity"]["video_id"]
            score   = hit["distance"]
            moved   = "▲" if vid_id not in stored_ids[:i] and vid_id in url_ids else ""
            snippet = _video_snippet(vid_id, cache_dir)
            print(f"  {i:2}. [{score:+.4f}]  {vid_id}")
            print(f"      {snippet}")

        print(f"\n{'─' * W}")
        ctx_label = "keywords" if keywords else "URL context"
        print(f"  [B] TOP {top_k} — recomputed with {ctx_label}")
        print(f"{'─' * W}")
        for i, hit in enumerate(url_hits, 1):
            vid_id  = hit["entity"]["video_id"]
            score   = hit["distance"]
            new_tag = "  ← NEW" if vid_id not in stored_ids else ""
            snippet = _video_snippet(vid_id, cache_dir)
            print(f"  {i:2}. [{score:+.4f}]  {vid_id}{new_tag}")
            print(f"      {snippet}")

        new_in_b = set(url_ids) - set(stored_ids)
        dropped  = set(stored_ids) - set(url_ids)
        print(f"\n  Page context swapped in {len(new_in_b)} new video(s), dropped {len(dropped)}")
    else:
        print(f"  TOP {top_k} RECOMMENDED VIDEOS")
        print(f"{'─' * W}")
        hits = _search(
            client, video_coll, source["embedding"],
            "video_id", ["brand_id"], top_k, brand_id,
        )
        for i, hit in enumerate(hits, 1):
            vid_id  = hit["entity"]["video_id"]
            score   = hit["distance"]
            snippet = _video_snippet(vid_id, cache_dir)
            print(f"  {i:2}. [{score:+.4f}]  {vid_id}")
            print(f"      {snippet}")

    print(f"{'=' * W}\n")


def run_user_to_video_rerank(
    client: MilvusClient,
    user_id: str,
    brand_id: int,
    top_k: int,
    video_coll: str,
    user_coll: str,
    cache_dir: Path,
    url: str | None,
    keywords: str | None,
    lam: float = _RERANK_LAMBDA,
) -> None:
    """
    Re-ranking mode (Variant C).
    Step 1 — ANN retrieval with static user embedding (no context in model).
    Step 2 — Re-rank using: λ * base_score + (1-λ) * dot(ctx_emb_768d, video_backbone_768d)
    Shows three lists: [A] base, [B] re-ranked, [C] new/dropped delta.
    """
    user_brand_key = f"{user_id}::{brand_id}"
    source  = _get_source(client, user_coll, "user_brand_key", user_brand_key, user_coll)
    persona = _user_persona(user_id, brand_id, source.get("cohort_ids", "0"), cache_dir)

    print('-' * 150)
    print(f"\n{'=' * W}")
    print(f"  USER PERSONA  [Variant C — re-ranking]")
    print(f"{'─' * W}")
    print(f"  User  : {user_id}")
    print(f"  Brand : {brand_id}")
    print(persona)

    # Resolve context text
    if keywords:
        ctx_text, ctx_source = keywords.strip(), "direct-keywords"
    elif url:
        ctx_text, ctx_source = _extract_url_context(url)
    else:
        ctx_text, ctx_source = "", "none"

    if not ctx_text:
        print(f"\n  [No context available — showing base ranking only]")
        print(f"\n{'=' * W}")
        print(f"  TOP {top_k} RECOMMENDED VIDEOS (base)")
        print(f"{'─' * W}")
        hits = _search(client, video_coll, source["embedding"],
                       "video_id", ["brand_id"], top_k, brand_id)
        for i, hit in enumerate(hits, 1):
            vid_id  = hit["entity"]["video_id"]
            snippet = _video_snippet(vid_id, cache_dir)
            print(f"  {i:2}. [{hit['distance']:+.4f}]  {vid_id}")
            print(f"      {snippet}")
        print(f"{'=' * W}\n")
        return

    print(f"\n{'─' * W}")
    label = f"URL  : {url}" if url else "Keywords : (direct)"
    print(f"  PAGE CONTEXT  ({ctx_source})")
    print(f"  {label}")
    print(f"  Text : {ctx_text[:200]}{'…' if len(ctx_text) > 200 else ''}")
    print(f"\n{'=' * W}")

    # Fetch a larger candidate pool for re-ranking (top_k * 5)
    pool_size  = min(top_k * 5, 500)
    pool_hits  = _search(client, video_coll, source["embedding"],
                         "video_id", ["brand_id"], pool_size, brand_id)
    pool_ids   = [h["entity"]["video_id"] for h in pool_hits]
    base_scores = {h["entity"]["video_id"]: h["distance"] for h in pool_hits}

    # Embed context and score against backbone video embeddings
    ctx_emb     = _embed_text(ctx_text)                         # 768d
    backbone_embs = _load_backbone_embs(pool_ids, cache_dir)    # {vid_id: 768d}

    # Re-rank: λ * normalised_base + (1-λ) * ctx_dot
    base_arr = np.array([base_scores[v] for v in pool_ids], dtype=np.float32)
    base_arr = (base_arr - base_arr.min()) / (base_arr.max() - base_arr.min() + 1e-8)

    ctx_norm  = ctx_emb / (np.linalg.norm(ctx_emb) + 1e-8)
    ctx_scores_arr = np.array([
        float(np.dot(backbone_embs[v] / (np.linalg.norm(backbone_embs[v]) + 1e-8), ctx_norm))
        if v in backbone_embs else 0.0
        for v in pool_ids
    ], dtype=np.float32)

    final_arr    = lam * base_arr + (1.0 - lam) * ctx_scores_arr
    rerank_order = [pool_ids[i] for i in np.argsort(-final_arr)]

    base_top    = pool_ids[:top_k]
    rerank_top  = rerank_order[:top_k]
    new_in_rerank = set(rerank_top) - set(base_top)
    dropped       = set(base_top) - set(rerank_top)

    print(f"  [A] TOP {top_k} — base ANN (no context)")
    print(f"{'─' * W}")
    for i, vid_id in enumerate(base_top, 1):
        snippet = _video_snippet(vid_id, cache_dir)
        print(f"  {i:2}. [{base_scores[vid_id]:+.4f}]  {vid_id}")
        print(f"      {snippet}")

    print(f"\n{'─' * W}")
    print(f"  [B] TOP {top_k} — re-ranked (λ={lam} base + {1-lam:.1f} context)")
    print(f"{'─' * W}")
    for i, vid_id in enumerate(rerank_top, 1):
        new_tag = "  ← NEW" if vid_id in new_in_rerank else ""
        bs      = base_scores.get(vid_id, 0.0)
        cs      = float(ctx_scores_arr[pool_ids.index(vid_id)])
        snippet = _video_snippet(vid_id, cache_dir)
        print(f"  {i:2}. [base={bs:+.4f} ctx={cs:+.4f}]  {vid_id}{new_tag}")
        print(f"      {snippet}")

    print(f"\n  Re-ranking swapped in {len(new_in_rerank)} new video(s), dropped {len(dropped)}")
    print(f"{'=' * W}\n")


def _load_backbone_embs(video_ids: list[str], cache_dir: Path) -> dict[str, np.ndarray]:
    """Load 768d backbone embeddings from cache for re-ranking ctx dot products."""
    emb_map: dict[str, np.ndarray] = _load_from_cache("video_embeddings", cache_dir) or {}
    return {v: emb_map[v] for v in video_ids if v in emb_map}


def _load_from_cache(name: str, cache_dir: Path):
    path = cache_dir / f"{name}.pkl"
    if path.exists():
        return pickle.load(open(path, "rb"))
    return None


def run_all_variants(
    client: MilvusClient,
    user_id: str,
    brand_id: int,
    top_k: int,
    url: str | None,
    keywords: str | None,
    lam: float = _RERANK_LAMBDA,
) -> None:
    """
    Runs user-to-video for all three trained variants with the same context signal
    and prints results side by side.

      [A] No context          — cache_web_no_ctx  / video_embeddings_web_no_ctx
      [B] Context in tower    — cache_web_ctx     / video_embeddings_web_ctx
      [C] Re-ranking at serve — cache_web_rerank  / video_embeddings_web_rerank
    """
    user_brand_key = f"{user_id}::{brand_id}"

    # Resolve page context text
    if keywords:
        ctx_text, ctx_source = keywords.strip(), "direct-keywords"
    elif url:
        ctx_text, ctx_source = _extract_url_context(url)
    else:
        ctx_text, ctx_source = "", "none"

    ctx_emb = _embed_text(ctx_text) if ctx_text else None   # 768d, used by B and C

    # User persona from whichever collection has this user
    persona_cache = _NOCTX_CACHE
    for src_coll, src_cache in [
        (_NOCTX_USER_COLL, _NOCTX_CACHE),
        (_RERANK_USER_COLL, _RERANK_CACHE),
        (_CTX_USER_COLL, _CTX_CACHE),
    ]:
        s = _try_get_source(client, src_coll, "user_brand_key", user_brand_key)
        if s:
            persona_cache = src_cache
            break

    print('-' * 150)
    print(f"\n{'=' * W}")
    print(f"  USER PERSONA")
    print(f"{'─' * W}")
    print(f"  User  : {user_id}")
    print(f"  Brand : {brand_id}")
    src_for_persona = _try_get_source(client, _NOCTX_USER_COLL, "user_brand_key", user_brand_key)
    cohort_ids_str  = (src_for_persona or {}).get("cohort_ids", "0")
    print(_user_persona(user_id, brand_id, cohort_ids_str, persona_cache))

    print(f"\n{'─' * W}")
    label = f"URL  : {url}" if url else "Keywords : (direct)"
    print(f"  PAGE CONTEXT  ({ctx_source})")
    print(f"  {label}")
    print(f"  Text : {ctx_text[:200]}{'…' if len(ctx_text) > 200 else ''}")
    print(f"{'=' * W}\n")

    results: dict[str, list[str]] = {"A": [], "B": [], "C": []}

    # ------------------------------------------------------------------
    # Variant A — no context
    # ------------------------------------------------------------------
    src_a = _try_get_source(client, _NOCTX_USER_COLL, "user_brand_key", user_brand_key)
    if src_a and client.has_collection(_NOCTX_VIDEO_COLL):
        hits_a = _search(client, _NOCTX_VIDEO_COLL, src_a["embedding"],
                         "video_id", ["brand_id"], top_k, brand_id)
        results["A"] = [h["entity"]["video_id"] for h in hits_a]
        print(f"  [A] Variant A — no context (stored embedding)")
        print(f"{'─' * W}")
        for i, hit in enumerate(hits_a, 1):
            vid_id  = hit["entity"]["video_id"]
            snippet = _video_snippet(vid_id, _NOCTX_CACHE)
            print(f"  {i:2}. [{hit['distance']:+.4f}]  {vid_id}")
            print(f"      {snippet}")
    else:
        print(f"  [A] Variant A — not available (run test_web.py first)")

    # ------------------------------------------------------------------
    # Variant B — context in tower (recompute with current ctx)
    # ------------------------------------------------------------------
    print(f"\n{'─' * W}")
    if client.has_collection(_CTX_VIDEO_COLL) and _CTX_MODEL_FILE.exists():
        if ctx_emb is not None:
            url_emb_b = _recompute_user_embedding(user_id, brand_id, ctx_emb, _CTX_CACHE)
            hits_b    = _search(client, _CTX_VIDEO_COLL, url_emb_b.tolist(),
                                "video_id", ["brand_id"], top_k, brand_id)
        else:
            src_b  = _try_get_source(client, _CTX_USER_COLL, "user_brand_key", user_brand_key)
            hits_b = _search(client, _CTX_VIDEO_COLL, src_b["embedding"],
                             "video_id", ["brand_id"], top_k, brand_id) if src_b else []
        results["B"] = [h["entity"]["video_id"] for h in hits_b]
        print(f"  [B] Variant B — context in tower (recomputed with current context)")
        print(f"{'─' * W}")
        for i, hit in enumerate(hits_b, 1):
            vid_id  = hit["entity"]["video_id"]
            new_tag = "  ← NEW vs A" if vid_id not in results["A"] else ""
            snippet = _video_snippet(vid_id, _CTX_CACHE)
            print(f"  {i:2}. [{hit['distance']:+.4f}]  {vid_id}{new_tag}")
            print(f"      {snippet}")
    else:
        print(f"  [B] Variant B — not available (run test_web_ctx.py first)")

    # ------------------------------------------------------------------
    # Variant C — re-ranking at serve time
    # ------------------------------------------------------------------
    print(f"\n{'─' * W}")
    src_c = _try_get_source(client, _RERANK_USER_COLL, "user_brand_key", user_brand_key)
    if src_c and client.has_collection(_RERANK_VIDEO_COLL):
        pool_size = min(top_k * 5, 500)
        pool_hits = _search(client, _RERANK_VIDEO_COLL, src_c["embedding"],
                            "video_id", ["brand_id"], pool_size, brand_id)
        pool_ids    = [h["entity"]["video_id"] for h in pool_hits]
        base_scores = {h["entity"]["video_id"]: h["distance"] for h in pool_hits}

        if ctx_emb is not None:
            backbone_embs = _load_backbone_embs(pool_ids, _RERANK_CACHE)
            base_arr  = np.array([base_scores[v] for v in pool_ids], dtype=np.float32)
            base_arr  = (base_arr - base_arr.min()) / (base_arr.max() - base_arr.min() + 1e-8)
            ctx_norm  = ctx_emb / (np.linalg.norm(ctx_emb) + 1e-8)
            ctx_arr   = np.array([
                float(np.dot(backbone_embs[v] / (np.linalg.norm(backbone_embs[v]) + 1e-8), ctx_norm))
                if v in backbone_embs else 0.0
                for v in pool_ids
            ], dtype=np.float32)
            final_arr    = lam * base_arr + (1.0 - lam) * ctx_arr
            rerank_order = [pool_ids[i] for i in np.argsort(-final_arr)]
            ctx_score_map = dict(zip(pool_ids, ctx_arr.tolist()))
        else:
            rerank_order  = pool_ids
            ctx_score_map = {}

        top_c = rerank_order[:top_k]
        results["C"] = top_c
        print(f"  [C] Variant C — re-ranking (λ={lam} base + {1 - lam:.1f} context)")
        print(f"{'─' * W}")
        for i, vid_id in enumerate(top_c, 1):
            bs      = base_scores.get(vid_id, 0.0)
            cs      = ctx_score_map.get(vid_id, 0.0)
            new_a   = " ← NEW vs A" if vid_id not in results["A"] else ""
            new_b   = " ← NEW vs B" if vid_id not in results["B"] else ""
            snippet = _video_snippet(vid_id, _RERANK_CACHE)
            print(f"  {i:2}. [base={bs:+.4f} ctx={cs:+.4f}]  {vid_id}{new_a}{new_b}")
            print(f"      {snippet}")
    else:
        print(f"  [C] Variant C — not available (run test_web_rerank.py first)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    filled = {k: v for k, v in results.items() if v}
    if len(filled) >= 2:
        sets = {k: set(v) for k, v in filled.items()}
        print(f"\n{'─' * W}")
        print(f"  OVERLAP SUMMARY (top-{top_k})")
        keys = list(sets.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = keys[i], keys[j]
                print(f"  {a}∩{b}: {len(sets[a] & sets[b])} shared  "
                      f"| unique to {a}: {len(sets[a] - sets[b])}  "
                      f"| unique to {b}: {len(sets[b] - sets[a])}")

    print(f"{'=' * W}\n")


def run_video_to_video(
    client: MilvusClient, video_id: str, top_k: int,
    video_coll: str, cache_dir: Path,
) -> None:
    source  = _get_source(client, video_coll, "video_id", video_id, "")
    snippet = _video_snippet(video_id, cache_dir, 200)

    print(f"\n{'=' * W}")
    print(f"  SOURCE VIDEO")
    print(f"{'─' * W}")
    print(f"  ID      : {video_id}")
    print(f"  Brand   : {source.get('brand_id')}")
    print(f"  Content : {snippet}")
    print(f"{'=' * W}")
    print(f"  TOP {top_k} SIMILAR VIDEOS")
    print(f"{'─' * W}")

    hits = _search(
        client, video_coll, source["embedding"],
        "video_id", ["brand_id"], top_k, exclude_id=video_id,
    )
    for i, hit in enumerate(hits, 1):
        vid_id  = hit["entity"]["video_id"]
        score   = hit["distance"]
        snippet = _video_snippet(vid_id, cache_dir)
        print(f"  {i:2}. [{score:+.4f}]  {vid_id}")
        print(f"      {snippet}")
    print(f"{'=' * W}\n")


def run_video_to_user(
    client: MilvusClient, video_id: str, top_k: int,
    video_coll: str, user_coll: str, cache_dir: Path,
) -> None:
    source  = _get_source(client, video_coll, "video_id", video_id, "")
    snippet = _video_snippet(video_id, cache_dir, 200)

    print(f"\n{'=' * W}")
    print(f"  SOURCE VIDEO")
    print(f"{'─' * W}")
    print(f"  ID      : {video_id}")
    print(f"  Content : {snippet}")
    print(f"{'=' * W}")
    print(f"  TOP {top_k} MATCHED USERS")
    print(f"{'─' * W}")

    hits = _search(
        client, user_coll, source["embedding"],
        "user_brand_key", ["user_id", "brand_id", "cohort_ids"], top_k,
    )
    for i, hit in enumerate(hits, 1):
        entity  = hit["entity"]
        score   = hit["distance"]
        cohorts = _cohort_label(entity.get("cohort_ids", "0"), cache_dir)
        print(f"  {i:2}. [{score:+.4f}]  user={entity['user_id']}  brand={entity['brand_id']}")
        print(f"      Cohorts: {cohorts}")
    print(f"{'=' * W}\n")


def run_user_to_user(
    client: MilvusClient, user_id: str, brand_id: int, top_k: int,
    user_coll: str, cache_dir: Path,
) -> None:
    user_brand_key = f"{user_id}::{brand_id}"
    source = _get_source(client, user_coll, "user_brand_key", user_brand_key, user_coll)
    persona = _user_persona(user_id, brand_id, source.get("cohort_ids", "0"), cache_dir)

    print(f"\n{'=' * W}")
    print(f"  SOURCE USER PERSONA")
    print(f"{'─' * W}")
    print(f"  User  : {user_id}")
    print(f"  Brand : {brand_id}")
    print(persona)
    print(f"{'=' * W}")
    print(f"  TOP {top_k} SIMILAR USERS")
    print(f"{'─' * W}")

    hits = _search(
        client, user_coll, source["embedding"],
        "user_brand_key", ["user_id", "brand_id", "cohort_ids"], top_k,
        exclude_id=user_brand_key,
    )
    for i, hit in enumerate(hits, 1):
        entity  = hit["entity"]
        score   = hit["distance"]
        cohorts = _cohort_label(entity.get("cohort_ids", "0"), cache_dir)
        print(f"  {i:2}. [{score:+.4f}]  user={entity['user_id']}  brand={entity['brand_id']}")
        print(f"      Cohorts: {cohorts}")
    print(f"{'=' * W}\n")


def run_list_cohorts(cache_dir: Path) -> None:
    profiles = _load("cohort_profiles", cache_dir)
    if not profiles:
        print(f"Error: cohort_profiles.pkl not found in {cache_dir}", file=sys.stderr)
        sys.exit(1)
    variant = "with page context" if cache_dir == _CTX_CACHE else "no page context"
    print(f"\n{'=' * W}")
    print(f"  COHORTS  ({len(profiles)} total)  [{variant}]")
    print(f"{'─' * W}")
    for p in sorted(profiles, key=lambda x: x["cohort_id"]):
        desc = p.get("description") or p.get("summary") or "(no description)"
        print(f"  [{p['cohort_id']}] {p['label']}")
        print(f"      {desc}")
    print(f"{'=' * W}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Web model inference with optional page context")
    parser.add_argument(
        "mode",
        choices=["user-to-video", "video-to-video", "video-to-user", "user-to-user", "list-cohorts"],
    )
    parser.add_argument("id",         type=str, nargs="?", help="Source user_id or video_id")
    parser.add_argument("--top-k",    type=int, default=20)
    parser.add_argument("--brand-id", type=int, default=None)
    parser.add_argument("--ctx",      action="store_true", help="Use page-context model (Variant B)")
    parser.add_argument("--rerank",   action="store_true", help="Use re-ranking model (Variant C)")
    parser.add_argument("--lambda",   type=float, default=_RERANK_LAMBDA, dest="lam",
                        help=f"Re-ranking weight on base score (default {_RERANK_LAMBDA}). "
                             "Only used with --rerank.")
    parser.add_argument("--url",      type=str, default=None,
                        help="Page URL to inject as real-time context")
    parser.add_argument("--keywords", type=str, default=None,
                        help="Comma-separated keywords (bypasses URL lookup). "
                             "Example: --keywords 'bmw, hybrid, convertible'")
    args = parser.parse_args()

    if args.ctx and args.rerank:
        parser.error("--ctx and --rerank are mutually exclusive")
    if args.url and args.keywords:
        parser.error("--url and --keywords are mutually exclusive")
    if (args.url or args.keywords) and args.mode != "user-to-video":
        parser.error("--url / --keywords only apply to user-to-video mode")

    # Select collections and cache dir for single-variant modes
    if args.rerank:
        cache_dir  = _RERANK_CACHE
        video_coll = _RERANK_VIDEO_COLL
        user_coll  = _RERANK_USER_COLL
    elif args.ctx:
        cache_dir  = _CTX_CACHE
        video_coll = _CTX_VIDEO_COLL
        user_coll  = _CTX_USER_COLL
    else:
        cache_dir  = _NOCTX_CACHE
        video_coll = _NOCTX_VIDEO_COLL
        user_coll  = _NOCTX_USER_COLL

    if args.mode == "list-cohorts":
        run_list_cohorts(cache_dir)
        return

    if args.id is None:
        parser.error(f"'id' is required for {args.mode} mode")
    if args.mode in ("user-to-video", "user-to-user") and args.brand_id is None:
        parser.error(f"--brand-id is required for {args.mode} mode")

    client = MilvusClient(uri=MILVUS_URI)

    if args.mode == "user-to-video":
        has_context = bool(args.url or args.keywords)
        if has_context and not args.ctx and not args.rerank:
            # URL/keywords without a variant flag → run all three variants
            run_all_variants(
                client, args.id, args.brand_id, args.top_k,
                url=args.url, keywords=args.keywords, lam=args.lam,
            )
        elif args.rerank:
            run_user_to_video_rerank(
                client, args.id, args.brand_id, args.top_k,
                video_coll, user_coll, cache_dir,
                url=args.url, keywords=args.keywords, lam=args.lam,
            )
        else:
            run_user_to_video(client, args.id, args.brand_id, args.top_k,
                              video_coll, user_coll, cache_dir, args.url,
                              keywords=args.keywords)
    elif args.mode == "video-to-video":
        run_video_to_video(client, args.id, args.top_k, video_coll, cache_dir)
    elif args.mode == "video-to-user":
        run_video_to_user(client, args.id, args.top_k, video_coll, user_coll, cache_dir)
    elif args.mode == "user-to-user":
        run_user_to_user(client, args.id, args.brand_id, args.top_k, user_coll, cache_dir)


main()
