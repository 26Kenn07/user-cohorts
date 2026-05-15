"""
Enhanced similarity queries — same modes as infer.py but with rich context:
  - user-to-video : user persona summary + description for each recommended video
  - video-to-video: source video description + description for each result
  - video-to-user : source video description + profile for each matched user
  - user-to-user  : source user persona + profile for each similar user
  - list-cohorts  : print all cohort IDs, labels, and descriptions

Usage:
    uv run infer_v2.py user-to-video  <user_id>   --brand-id <id>  [--top-k 20]
    uv run infer_v2.py video-to-video <video_id>                   [--top-k 20]
    uv run infer_v2.py video-to-user  <video_id>                   [--top-k 20]
    uv run infer_v2.py user-to-user   <user_id>   --brand-id <id>  [--top-k 20]
    uv run infer_v2.py list-cohorts
"""

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

from dotenv import load_dotenv
from pymilvus import MilvusClient

load_dotenv()
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

MILVUS_URI       = os.environ.get("MILVUS_URI", "http://localhost:19530")
VIDEO_COLLECTION = os.environ.get("VIDEO_COLLECTION", "video_embeddings")
USER_COLLECTION  = os.environ.get("USER_COLLECTION", "user_embeddings")
CACHE_DIR        = Path(os.environ.get("CACHE_DIR", "./cache"))
DESC_WIDTH       = 100   # characters to show per video description
W                = 70    # output separator width


# ---------------------------------------------------------------------------
# Cache loaders
# ---------------------------------------------------------------------------

_cache: dict = {}

def _load(name: str):
    if name not in _cache:
        path = CACHE_DIR / f"{name}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                _cache[name] = pickle.load(f)
        else:
            _cache[name] = None
    return _cache[name]


def _video_lookup() -> dict[str, dict]:
    videos = _load("videos") or []
    return {v["video_id"]: v for v in videos}


def _cohort_lookup() -> dict[int, dict]:
    profiles = _load("cohort_profiles") or []
    return {p["cohort_id"]: p for p in profiles}


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


def _search(
    client: MilvusClient,
    target_collection: str,
    embedding: list,
    target_id_field: str,
    target_extra: list[str],
    top_k: int,
    brand_id: int | None,
    exclude_id: str | None,
) -> list[dict]:
    filter_expr = f"brand_id == {brand_id}" if brand_id is not None else ""
    limit = top_k + 1 if exclude_id else top_k
    results = client.search(
        collection_name=target_collection,
        data=[embedding],
        anns_field="embedding",
        limit=limit,
        output_fields=[target_id_field] + target_extra,
        filter=filter_expr,
        search_params={"metric_type": "IP", "params": {"nprobe": 128}},
    )[0]
    if exclude_id:
        results = [h for h in results if h["entity"][target_id_field] != exclude_id]
    return results[:top_k]


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

def _video_snippet(video_id: str, width: int = DESC_WIDTH) -> str:
    meta = _video_lookup().get(video_id, {})
    def _clean(val: object) -> str:
        s = str(val).strip() if val else ""
        return "" if s.lower() in ("nan", "none", "") else s
    text = (
        _clean(meta.get("video_gen_description")) or
        _clean(meta.get("description_text")) or
        _clean(meta.get("transcript")) or ""
    ).replace("\n", " ")
    if not text:
        return "(no description available)"
    return text[:width] + ("…" if len(text) > width else "")


def _cohort_label(cohort_ids_str: str) -> str:
    by_id = _cohort_lookup()
    ids = [int(x) for x in cohort_ids_str.split(",") if x.strip().isdigit()]
    labels = [by_id[cid]["label"] for cid in ids if cid in by_id]
    return ", ".join(labels) if labels else cohort_ids_str


def _user_persona(user_id: str, brand_id: int, cohort_ids_str: str) -> str:
    """Builds a persona summary from cached engagement data."""
    import numpy as np

    train_df = _load("train_df")
    if train_df is None:
        return "(engagement data not available)"

    g = train_df[
        (train_df["user_id"].astype(str) == str(user_id)) &
        (train_df["brand_id"].astype(str) == str(brand_id))
    ]
    if g.empty:
        return "(user not found in training data)"

    n_videos      = len(g)
    avg_watch     = g["watch_percentage"].mean()
    total_likes   = int(g["likes"].sum())
    total_shares  = int(g["shares"].sum())
    total_comments= int(g["comments"].sum())

    # Engagement style
    if total_shares > 0:
        style = f"active sharer ({total_shares} shares)"
    elif total_likes > 0:
        style = f"liker ({total_likes} likes)"
    elif avg_watch >= 70:
        style = "deep watcher (no explicit reactions)"
    else:
        style = "passive browser"

    # Top 3 most-engaged videos
    g = g.copy()
    g["_score"] = (
        (g["watch_percentage"] / 100) * 1.0
        + g["views"].clip(upper=3) * 0.5
        + g["likes"] * 3.0
        + g["shares"] * 5.0
        + g["comments"] * 3.0
    )
    top_vids = g.nlargest(3, "_score")["video_id"].tolist()
    vl = _video_lookup()
    top_snippets = []
    for vid in top_vids:
        snip = _video_snippet(vid, width=70)
        top_snippets.append(f"    · {snip}")

    cohort_str = _cohort_label(cohort_ids_str)
    lines = [
        f"  Videos watched  : {n_videos}",
        f"  Avg watch time  : {avg_watch:.1f}%",
        f"  Reactions       : {total_likes} likes · {total_shares} shares · {total_comments} comments",
        f"  Engagement style: {style}",
        f"  Content cohorts : {cohort_str}",
        f"  Top engaged videos:",
    ] + (top_snippets if top_snippets else ["    (none)"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------

def run_user_to_video(client: MilvusClient, user_id: str, brand_id: int, top_k: int) -> None:
    user_brand_key = f"{user_id}::{brand_id}"
    source = _get_source(client, USER_COLLECTION, "user_brand_key", user_brand_key)

    cohort_ids_str = source.get("cohort_ids", "0")
    persona = _user_persona(user_id, brand_id, cohort_ids_str)

    print(f"\n{'=' * W}")
    print(f"  USER PERSONA")
    print(f"{'─' * W}")
    print(f"  User    : {user_id}")
    print(f"  Brand   : {brand_id}")
    print(persona)
    print(f"{'=' * W}")
    print(f"  TOP {top_k} RECOMMENDED VIDEOS")
    print(f"{'─' * W}")

    hits = _search(
        client, VIDEO_COLLECTION, source["embedding"],
        "video_id", ["brand_id", "has_engagement"],
        top_k, brand_id, exclude_id=None,
    )
    for i, hit in enumerate(hits, 1):
        vid_id  = hit["entity"]["video_id"]
        score   = hit["distance"]
        # eng_tag = "✓ engaged" if hit["entity"]["has_engagement"] else "  new"
        snippet = _video_snippet(vid_id)
        print(f"  {i:2}. [{score:+.4f}  {vid_id}")
        print(f"      {snippet}")
    print(f"{'=' * W}\n")


def run_video_to_video(client: MilvusClient, video_id: str, top_k: int) -> None:
    source  = _get_source(client, VIDEO_COLLECTION, "video_id", video_id)
    snippet = _video_snippet(video_id, width=200)

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
        client, VIDEO_COLLECTION, source["embedding"],
        "video_id", ["brand_id", "has_engagement"],
        top_k, brand_id=None, exclude_id=video_id,
    )
    for i, hit in enumerate(hits, 1):
        vid_id  = hit["entity"]["video_id"]
        score   = hit["distance"]
        snippet = _video_snippet(vid_id)
        print(f"  {i:2}. [{score:+.4f}]  {vid_id}")
        print(f"      {snippet}")
    print(f"{'=' * W}\n")


def run_video_to_user(client: MilvusClient, video_id: str, top_k: int) -> None:
    source  = _get_source(client, VIDEO_COLLECTION, "video_id", video_id)
    snippet = _video_snippet(video_id, width=200)

    print(f"\n{'=' * W}")
    print(f"  SOURCE VIDEO")
    print(f"{'─' * W}")
    print(f"  ID      : {video_id}")
    print(f"  Content : {snippet}")
    print(f"{'=' * W}")
    print(f"  TOP {top_k} MATCHED USERS")
    print(f"{'─' * W}")

    hits = _search(
        client, USER_COLLECTION, source["embedding"],
        "user_brand_key", ["user_id", "brand_id", "cohort_ids"],
        top_k, brand_id=None, exclude_id=None,
    )
    for i, hit in enumerate(hits, 1):
        entity  = hit["entity"]
        score   = hit["distance"]
        cohorts = _cohort_label(entity.get("cohort_ids", "0"))
        print(f"  {i:2}. [{score:+.4f}]  user={entity['user_id']}  brand={entity['brand_id']}")
        print(f"      Cohorts: {cohorts}")
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

    hits = _search(
        client, USER_COLLECTION, source["embedding"],
        "user_brand_key", ["user_id", "brand_id", "cohort_ids"],
        top_k, brand_id=None, exclude_id=user_brand_key,
    )
    for i, hit in enumerate(hits, 1):
        entity  = hit["entity"]
        score   = hit["distance"]
        cohorts = _cohort_label(entity.get("cohort_ids", "0"))
        print(f"  {i:2}. [{score:+.4f}]  user={entity['user_id']}  brand={entity['brand_id']}")
        print(f"      Cohorts: {cohorts}")
    print(f"{'=' * W}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Enhanced Milvus similarity queries with context")
    parser.add_argument(
        "mode",
        choices=["user-to-video", "video-to-video", "video-to-user", "user-to-user", "list-cohorts"],
    )
    parser.add_argument("id",          type=str, nargs="?",   help="Source user_id or video_id (not needed for list-cohorts)")
    parser.add_argument("--top-k",     type=int, default=20)
    parser.add_argument("--brand-id",  type=int, default=None)
    args = parser.parse_args()

    if args.mode == "list-cohorts":
        run_list_cohorts()
        return

    if args.id is None:
        parser.error(f"'id' is required for {args.mode} mode")
    if args.mode in ("user-to-video", "user-to-user") and args.brand_id is None:
        parser.error(f"--brand-id is required for {args.mode} mode")

    client = MilvusClient(uri=MILVUS_URI)

    if args.mode == "user-to-video":
        run_user_to_video(client, args.id, args.brand_id, args.top_k)
    elif args.mode == "video-to-video":
        run_video_to_video(client, args.id, args.top_k)
    elif args.mode == "video-to-user":
        run_video_to_user(client, args.id, args.top_k)
    elif args.mode == "user-to-user":
        run_user_to_user(client, args.id, args.brand_id, args.top_k)


main()
