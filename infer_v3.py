"""
Page-context-aware recommendation inference.

Modes:
  user-to-video  : personalised video feed, optionally steered by page context
  video-to-video : similar videos to a given video
  video-to-user  : users most likely to engage with a video
  user-to-user   : users with similar taste
  list-cohorts   : print all cohort IDs, labels, and descriptions

Without --page-context  →  standard ANN search (same as infer_v2).

With --page-context "BMW luxury sedan performance"  →  hybrid path:
  1. Embed page keywords via backbone → VideoTower → 512d  (ctx_emb)
  2. query_emb = normalize(CTX_ALPHA × ctx_emb + (1-CTX_ALPHA) × user_emb)
     Blending pulls the ANN query toward page-relevant content so the
     candidate pool is not purely driven by past user history.
  3. ANN in Milvus with query_emb → top ANN_CANDIDATES (default 200)
  4. BM25 in OpenSearch, filtered to those candidate IDs,
     queried against transcript / descriptions / keywords
  5. RRF(ANN_rank, BM25_rank, k=60) → final top-k

Usage:
    uv run infer_v3.py user-to-video  <user_id> --brand-id <id> [--top-k 20] [--page-context "..."]
    uv run infer_v3.py video-to-video <video_id>                [--top-k 20]
    uv run infer_v3.py video-to-user  <video_id>                [--top-k 20]
    uv run infer_v3.py user-to-user   <user_id> --brand-id <id> [--top-k 20]
    uv run infer_v3.py list-cohorts
"""

import argparse
import asyncio
import logging
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from pymilvus import MilvusClient

load_dotenv()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

MILVUS_URI       = os.environ.get("MILVUS_URI", "http://localhost:19530")
VIDEO_COLLECTION = os.environ.get("VIDEO_COLLECTION", "cohort_video_embeddings")
USER_COLLECTION  = os.environ.get("USER_COLLECTION",  "cohort_user_embeddings")
CACHE_DIR        = Path(os.environ.get("CACHE_DIR", "./cache"))
MODEL_PATH       = CACHE_DIR / "two_tower.pt"
OS_INDEX         = "genuin_loop_video_index"

ANN_CANDIDATES = 500   # candidates fetched from Milvus before re-ranking
RRF_K          = 60    # RRF constant (Cormack et al. 2009)
CTX_ALPHA      = 0.3   # weight given to page context; user gets (1 - CTX_ALPHA)
DESC_WIDTH     = 200
W              = 70

LOG_FILE = Path("infer_v3.log")

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
# Cache helpers
# ---------------------------------------------------------------------------

_cache: dict = {}

def _load(name: str):
    if name not in _cache:
        path = CACHE_DIR / f"{name}.pkl"
        _cache[name] = pickle.load(open(path, "rb")) if path.exists() else None
    return _cache[name]

def _video_lookup() -> dict[str, dict]:
    return {v["video_id"]: v for v in (_load("videos") or [])}

def _cohort_lookup() -> dict[int, dict]:
    return {p["cohort_id"]: p for p in (_load("cohort_profiles") or [])}


# ---------------------------------------------------------------------------
# VideoTower loader  (used to embed page context into 512d space)
# ---------------------------------------------------------------------------

_video_tower = None

def _get_video_tower():
    global _video_tower
    if _video_tower is not None:
        return _video_tower
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {MODEL_PATH} — run test.py first")
    from models.two_tower import VideoTower
    state = torch.load(MODEL_PATH, map_location="cpu")
    tower_state = {k.replace("video_tower.", ""): v for k, v in state.items() if k.startswith("video_tower.")}
    tower = VideoTower(backbone_dim=768, output_dim=512)
    tower.load_state_dict(tower_state)
    tower.eval()
    _video_tower = tower
    return _video_tower


# ---------------------------------------------------------------------------
# Page context embedding
# ---------------------------------------------------------------------------

def embed_page_context(text: str) -> np.ndarray:
    """
    Embeds page context keywords into the same 512d space as user/video embeddings.
    Pipeline: text → backbone (768d) → VideoTower MLP → 512d L2-normalised.
    """
    from utils.embeddings import _get_model
    backbone = _get_model()
    raw = backbone.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
    tower = _get_video_tower()
    with torch.no_grad():
        t = torch.tensor(raw, dtype=torch.float32).unsqueeze(0)
        ctx_emb = tower(t).squeeze(0).cpu().numpy()
    return ctx_emb  # already L2-normalised by VideoTower


def blend_embeddings(user_emb: np.ndarray, ctx_emb: np.ndarray, alpha: float = CTX_ALPHA) -> list:
    """
    Blends user and context embeddings, re-normalises, returns as list for Milvus.
    alpha controls the page-context weight; user gets (1 - alpha).
    """
    blended = (1.0 - alpha) * user_emb + alpha * ctx_emb
    norm = np.linalg.norm(blended)
    if norm > 0:
        blended /= norm
    return blended.tolist()


# ---------------------------------------------------------------------------
# OpenSearch BM25
# ---------------------------------------------------------------------------

async def _bm25_scores_async(candidate_ids: list[str], query: str) -> dict[str, float]:
    from opensearchpy import AsyncOpenSearch
    from utils.config import settings

    client = AsyncOpenSearch(
        hosts=[settings.opensearch.url],
        http_auth=(settings.opensearch.user_name, settings.opensearch.password),
        verify_certs=settings.opensearch.os_verify,
        use_ssl=True,
        ssl_show_warn=False,
        timeout=30,
    )
    try:
        resp = await client.search(
            index=OS_INDEX,
            body={
                "query": {
                    "bool": {
                        "filter": [{"ids": {"values": candidate_ids}}],
                        "should": [
                            {"match": {"transcript":                 {"query": query, "boost": 4.0}}},
                            {"match": {"description_text":           {"query": query, "boost": 3.0}}},
                            {"match": {"video_gen_description":      {"query": query, "boost": 2.5}}},
                            {"match": {"keywords.explicit.keyword":  {"query": query, "boost": 0.5}}},
                        ],
                        "minimum_should_match": 0,
                    }
                },
                "_source": False,
                "size": len(candidate_ids),
            },
        )
        return {hit["_id"]: float(hit["_score"] or 0.0) for hit in resp["hits"]["hits"]}
    finally:
        await client.close()


def bm25_scores(candidate_ids: list[str], query: str) -> dict[str, float]:
    return asyncio.run(_bm25_scores_async(candidate_ids, query))


# ---------------------------------------------------------------------------
# RRF
# ---------------------------------------------------------------------------

def rrf_combine(
    ann_hits: list[dict],
    bm25_map: dict[str, float],
    top_k: int,
    k: int = RRF_K,
) -> list[dict]:
    """
    Reciprocal Rank Fusion over ANN and BM25 rankings.
    Returns top_k hits re-ordered by combined RRF score.
    """
    scores: dict[str, float] = {}

    for rank, hit in enumerate(ann_hits, 1):
        vid = hit["entity"]["video_id"]
        scores[vid] = scores.get(vid, 0.0) + 1.0 / (k + rank)

    if bm25_map:
        bm25_ranked = sorted(bm25_map.items(), key=lambda x: x[1], reverse=True)
        for rank, (vid, _) in enumerate(bm25_ranked, 1):
            if vid in scores:
                scores[vid] += 1.0 / (k + rank)

    hit_by_id = {h["entity"]["video_id"]: h for h in ann_hits}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"entity": hit_by_id[vid]["entity"], "distance": rrf_score, "rrf": True}
            for vid, rrf_score in ranked if vid in hit_by_id]


# ---------------------------------------------------------------------------
# Milvus helpers
# ---------------------------------------------------------------------------

def _get_source(client: MilvusClient, collection: str, id_field: str, id_value: str) -> dict:
    extra = ["user_id", "brand_id", "cohort_ids"] if collection == USER_COLLECTION \
            else ["brand_id", "has_engagement"]
    rows = client.query(
        collection_name=collection,
        filter=f'{id_field} == "{id_value}"',
        output_fields=["embedding"] + extra,
    )
    if not rows:
        print(f"Error: '{id_value}' not found in '{collection}'", file=sys.stderr)
        sys.exit(1)
    return rows[0]


def _ann_search(
    client: MilvusClient,
    target_collection: str,
    embedding: list,
    target_id_field: str,
    target_extra: list[str],
    limit: int,
    brand_id: int | None,
    exclude_id: str | None,
) -> list[dict]:
    filter_expr = f"brand_id == {brand_id}" if brand_id is not None else ""
    fetch = limit + 1 if exclude_id else limit
    results = client.search(
        collection_name=target_collection,
        data=[embedding],
        anns_field="embedding",
        limit=fetch,
        output_fields=[target_id_field] + target_extra,
        filter=filter_expr,
        search_params={"metric_type": "IP", "params": {"nprobe": 128}},
    )[0]
    if exclude_id:
        results = [h for h in results if h["entity"][target_id_field] != exclude_id]
    return results[:limit]


# ---------------------------------------------------------------------------
# Formatters  (same as infer_v2)
# ---------------------------------------------------------------------------

def _video_snippet(video_id: str, width: int = DESC_WIDTH) -> str:
    meta = _video_lookup().get(video_id, {})
    def _clean(val) -> str:
        s = str(val).strip() if val else ""
        return "" if s.lower() in ("nan", "none", "") else s
    text = (
        _clean(meta.get("video_gen_description")) or
        _clean(meta.get("description_text")) or
        _clean(meta.get("transcript")) or ""
    ).replace("\n", " ")
    return (text[:width] + "…" if len(text) > width else text) if text else "(no description)"


def _cohort_label(cohort_ids_str: str) -> str:
    by_id = _cohort_lookup()
    ids = [int(x) for x in cohort_ids_str.split(",") if x.strip().isdigit()]
    labels = [by_id[cid]["label"] for cid in ids if cid in by_id]
    return ", ".join(labels) if labels else cohort_ids_str


def _user_persona(user_id: str, brand_id: int, cohort_ids_str: str) -> str:
    train_df = _load("train_df")
    if train_df is None:
        return "(engagement data not available)"
    g = train_df[
        (train_df["user_id"].astype(str) == str(user_id)) &
        (train_df["brand_id"].astype(str) == str(brand_id))
    ]
    if g.empty:
        return "(user not found in training data)"
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
        + g["likes"] * 3.0 + g["shares"] * 5.0 + g["comments"] * 3.0
    )
    top_vids = g.nlargest(3, "_score")["video_id"].tolist()
    top_snippets = [f"    · {_video_snippet(v)}" for v in top_vids]
    return "\n".join([
        f"  Videos watched  : {len(g)}",
        f"  Avg watch time  : {avg_watch:.1f}%",
        f"  Reactions       : {total_likes} likes · {total_shares} shares · {total_comments} comments",
        f"  Engagement style: {style}",
        f"  Content cohorts : {_cohort_label(cohort_ids_str)}",
        f"  Top engaged videos:",
    ] + (top_snippets or ["    (none)"]))


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------

def run_user_to_video(
    client: MilvusClient,
    user_id: str,
    brand_id: int,
    top_k: int,
    page_context: str | None,
) -> None:
    user_brand_key = f"{user_id}::{brand_id}"
    source = _get_source(client, USER_COLLECTION, "user_brand_key", user_brand_key)
    user_emb = np.array(source["embedding"], dtype=np.float32)
    cohort_ids_str = source.get("cohort_ids", "0")

    # --- Pure ANN (no context) ---
    hits_base = _ann_search(
        client, VIDEO_COLLECTION, user_emb.tolist(),
        "video_id", ["brand_id", "has_engagement"],
        top_k, brand_id, exclude_id=None,
    )

    # --- User persona header ---
    persona = _user_persona(user_id, brand_id, cohort_ids_str)
    mode_tag = f"hybrid (ANN×{ANN_CANDIDATES} → BM25 → RRF)" if page_context else "ANN only"
    print('─'*220)
    print(f"\n{'=' * W}")
    print(f"  USER PERSONA  [{mode_tag}]")
    print(f"{'─' * W}")
    print(f"  User    : {user_id}")
    print(f"  Brand   : {brand_id}")
    print(persona)
    if page_context:
        print(f"  Page ctx: {page_context}")
    print(f"{'=' * W}")

    if page_context:
        # --- Hybrid path: blended ANN → BM25 → RRF ---
        print(f"\n  [page context] embedding: \"{page_context}\"")
        ctx_emb = embed_page_context(page_context)
        query   = blend_embeddings(user_emb, ctx_emb, CTX_ALPHA)
        hits_ctx = _ann_search(
            client, VIDEO_COLLECTION, query,
            "video_id", ["brand_id", "has_engagement"],
            ANN_CANDIDATES, brand_id, exclude_id=None,
        )
        if hits_ctx:
            candidate_ids = [h["entity"]["video_id"] for h in hits_ctx]
            bm25_map  = bm25_scores(candidate_ids, page_context)
            hits_ctx  = rrf_combine(hits_ctx, bm25_map, top_k)

        # --- Side-by-side output ---
        print(f"{'─' * W}")
        print(f"  {'WITHOUT page context (ANN)':<34}  {'WITH page context (RRF)'}")
        print(f"{'─' * W}")
        for i in range(top_k):
            left  = hits_base[i] if i < len(hits_base) else None
            right = hits_ctx[i]  if i < len(hits_ctx)  else None
            l_id  = left["entity"]["video_id"]  if left  else ""
            r_id  = right["entity"]["video_id"] if right else ""
            l_sc  = f"{left['distance']:+.4f}"  if left  else ""
            r_sc  = f"{right['distance']:+.4f}" if right else ""
            print(f"  {i+1:2}. [{l_sc}] {l_id:<28}  [{r_sc}] {r_id}")
        print(f"{'─' * W}")
        print(f"  DETAILS — WITHOUT context")
        print(f"{'─' * W}")
        for i, hit in enumerate(hits_base[:top_k], 1):
            vid_id = hit["entity"]["video_id"]
            print(f"  {i:2}. [score={hit['distance']:+.4f}]  {vid_id}")
            print(f"      {_video_snippet(vid_id)}")
        print(f"{'─' * W}")
        print(f"  DETAILS — WITH context  (\"{page_context}\")")
        print(f"{'─' * W}")
        for i, hit in enumerate(hits_ctx[:top_k], 1):
            vid_id = hit["entity"]["video_id"]
            print(f"  {i:2}. [RRF={hit['distance']:+.4f}]  {vid_id}")
            print(f"      {_video_snippet(vid_id)}")
    else:
        print(f"  TOP {top_k} RECOMMENDED VIDEOS")
        print(f"{'─' * W}")
        for i, hit in enumerate(hits_base[:top_k], 1):
            vid_id = hit["entity"]["video_id"]
            print(f"  {i:2}. [score={hit['distance']:+.4f}]  {vid_id}")
            print(f"      {_video_snippet(vid_id)}")

    print(f"{'=' * W}\n")


def run_video_to_video(client: MilvusClient, video_id: str, top_k: int) -> None:
    source  = _get_source(client, VIDEO_COLLECTION, "video_id", video_id)
    snippet = _video_snippet(video_id)
    print(f"\n{'=' * W}")
    print(f"  SOURCE VIDEO")
    print(f"{'─' * W}")
    print(f"  ID      : {video_id}")
    print(f"  Brand   : {source.get('brand_id')}")
    print(f"  Content : {snippet}")
    print(f"{'=' * W}")
    print(f"  TOP {top_k} SIMILAR VIDEOS")
    print(f"{'─' * W}")
    hits = _ann_search(client, VIDEO_COLLECTION, source["embedding"],
                       "video_id", ["brand_id", "has_engagement"], top_k, None, video_id)
    for i, hit in enumerate(hits, 1):
        vid_id = hit["entity"]["video_id"]
        print(f"  {i:2}. [{hit['distance']:+.4f}]  {vid_id}")
        print(f"      {_video_snippet(vid_id)}")
    print(f"{'=' * W}\n")


def run_video_to_user(client: MilvusClient, video_id: str, top_k: int) -> None:
    source  = _get_source(client, VIDEO_COLLECTION, "video_id", video_id)
    snippet = _video_snippet(video_id)
    print(f"\n{'=' * W}")
    print(f"  SOURCE VIDEO")
    print(f"{'─' * W}")
    print(f"  ID      : {video_id}")
    print(f"  Content : {snippet}")
    print(f"{'=' * W}")
    print(f"  TOP {top_k} MATCHED USERS")
    print(f"{'─' * W}")
    hits = _ann_search(client, USER_COLLECTION, source["embedding"],
                       "user_brand_key", ["user_id", "brand_id", "cohort_ids"], top_k, None, None)
    for i, hit in enumerate(hits, 1):
        e = hit["entity"]
        print(f"  {i:2}. [{hit['distance']:+.4f}]  user={e['user_id']}  brand={e['brand_id']}")
        print(f"      Cohorts: {_cohort_label(e.get('cohort_ids', '0'))}")
    print(f"{'=' * W}\n")


def run_user_to_user(client: MilvusClient, user_id: str, brand_id: int, top_k: int) -> None:
    user_brand_key = f"{user_id}::{brand_id}"
    source = _get_source(client, USER_COLLECTION, "user_brand_key", user_brand_key)
    cohort_ids_str = source.get("cohort_ids", "0")
    persona = _user_persona(user_id, brand_id, cohort_ids_str)
    print(f"\n{'=' * W}")
    print(f"  SOURCE USER PERSONA")
    print(f"{'─' * W}")
    print(f"  User    : {user_id}")
    print(f"  Brand   : {brand_id}")
    print(persona)
    print(f"{'=' * W}")
    print(f"  TOP {top_k} SIMILAR USERS")
    print(f"{'─' * W}")
    hits = _ann_search(client, USER_COLLECTION, source["embedding"],
                       "user_brand_key", ["user_id", "brand_id", "cohort_ids"], top_k, None, user_brand_key)
    for i, hit in enumerate(hits, 1):
        e = hit["entity"]
        print(f"  {i:2}. [{hit['distance']:+.4f}]  user={e['user_id']}  brand={e['brand_id']}")
        print(f"      Cohorts: {_cohort_label(e.get('cohort_ids', '0'))}")
    print(f"{'=' * W}\n")


def run_list_cohorts() -> None:
    profiles = _load("cohort_profiles")
    if not profiles:
        print("Error: cache/cohort_profiles.pkl not found — run test.py first.", file=sys.stderr)
        sys.exit(1)
    print(f"\n{'=' * W}")
    print(f"  COHORTS  ({len(profiles)} total)")
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
    parser = argparse.ArgumentParser(description="Page-context-aware recommendation inference")
    parser.add_argument("mode", choices=["user-to-video", "video-to-video", "video-to-user",
                                         "user-to-user", "list-cohorts"])
    parser.add_argument("id",            type=str,  nargs="?", help="Source user_id or video_id")
    parser.add_argument("--brand-id",    type=int,  default=None)
    parser.add_argument("--top-k",       type=int,  default=20)
    parser.add_argument("--page-context",type=str,  default=None,
                        help="Page context keywords to steer recommendations (user-to-video only)")
    parser.add_argument("--ctx-alpha",   type=float, default=CTX_ALPHA,
                        help=f"Page context weight in query blend (default {CTX_ALPHA})")
    args = parser.parse_args()

    if args.mode == "list-cohorts":
        run_list_cohorts()
        return

    if args.id is None:
        parser.error(f"'id' is required for {args.mode} mode")
    if args.mode in ("user-to-video", "user-to-user") and args.brand_id is None:
        parser.error(f"--brand-id is required for {args.mode} mode")
    if args.page_context and args.mode != "user-to-video":
        parser.error("--page-context is only supported for user-to-video mode")

    client = MilvusClient(uri=MILVUS_URI)

    if args.mode == "user-to-video":
        run_user_to_video(client, args.id, args.brand_id, args.top_k, args.page_context)
    elif args.mode == "video-to-video":
        run_video_to_video(client, args.id, args.top_k)
    elif args.mode == "video-to-user":
        run_video_to_user(client, args.id, args.top_k)
    elif args.mode == "user-to-user":
        run_user_to_user(client, args.id, args.brand_id, args.top_k)


main()
