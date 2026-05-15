"""
Personalised suggested prompts via Gemini.

Usage:
    # Prompts for a user (based on their cohort interests)
    uv run suggest.py user <user_id> --brand-id <brand_id>

    # Prompts based on a video's content
    uv run suggest.py video <video_id>

    # Prompts blending user interest + the video they're currently watching
    uv run suggest.py user-video <user_id> --brand-id <brand_id> --video-id <video_id>
"""

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from pymilvus import MilvusClient

load_dotenv()

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

MILVUS_URI       = os.environ.get("MILVUS_URI", "http://localhost:19530")
VIDEO_COLLECTION = os.environ.get("VIDEO_COLLECTION", "video_embeddings")
USER_COLLECTION  = os.environ.get("USER_COLLECTION", "user_embeddings")
CACHE_DIR        = Path(os.environ.get("CACHE_DIR", "./cache"))
N_PROMPTS        = 5

_gemini_client: genai.Client | None = None


def _gemini() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not set")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _load_cache(name: str) -> Any:
    path = CACHE_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Cache not found: {path} — run test.py first")
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Milvus helpers
# ---------------------------------------------------------------------------

def _query_user(client: MilvusClient, user_brand_key: str) -> dict:
    rows = client.query(
        collection_name=USER_COLLECTION,
        filter=f'user_brand_key == "{user_brand_key}"',
        output_fields=["user_id", "brand_id", "cohort_ids"],
    )
    if not rows:
        raise ValueError(f"User '{user_brand_key}' not found in Milvus")
    return rows[0]


def _query_video_exists(client: MilvusClient, video_id: str) -> None:
    rows = client.query(
        collection_name=VIDEO_COLLECTION,
        filter=f'video_id == "{video_id}"',
        output_fields=["video_id"],
    )
    if not rows:
        raise ValueError(f"Video '{video_id}' not found in Milvus")


# ---------------------------------------------------------------------------
# Video metadata — local cache first, OpenSearch fallback
# ---------------------------------------------------------------------------

_video_cache: dict[str, dict] | None = None


def _get_video_meta(video_id: str) -> dict:
    global _video_cache
    if _video_cache is None:
        path = CACHE_DIR / "videos.pkl"
        if path.exists():
            import pickle
            with open(path, "rb") as f:
                videos: list[dict] = pickle.load(f)
            _video_cache = {v["video_id"]: v for v in videos}
        else:
            _video_cache = {}

    if video_id in _video_cache:
        return _video_cache[video_id]

    # Fallback: hit OpenSearch (only reachable inside Docker)
    try:
        import asyncio
        from db.opensearch import get_videos_by_ids
        results = asyncio.run(get_videos_by_ids([video_id]))
        return results[0] if results else {}
    except Exception as e:
        logging.warning(f"OpenSearch unreachable, no metadata for {video_id}: {e}")
        return {}


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------

def _cohort_context(cohort_ids_str: str, profiles: list[dict]) -> str:
    """Converts a user's comma-separated cohort IDs into human-readable interest text."""
    by_id = {p["cohort_id"]: p for p in profiles}
    ids   = [int(x) for x in cohort_ids_str.split(",") if x.strip().isdigit()]

    lines = []
    for cid in ids:
        p = by_id.get(cid)
        if not p:
            continue
        label       = p.get("label", f"Cohort {cid}")
        description = p.get("description", "")
        keywords    = ", ".join(p.get("top_keywords", [])[:10])
        lines.append(f"  • {label}: {description}")
        if keywords:
            lines.append(f"    Top topics: {keywords}")

    return "\n".join(lines) if lines else "  • (unknown cohort)"


def _video_context(meta: dict) -> str:
    """Converts video metadata into a text summary for the prompt."""
    snippets = []
    for field in ("video_gen_description", "description_text", "transcript"):
        text = (meta.get(field) or "").strip()
        if text:
            snippets.append(text[:400])
            break

    keywords = meta.get("keywords") or []
    if keywords:
        snippets.append(f"Keywords: {', '.join(keywords[:15])}")

    return "\n".join(snippets) if snippets else "(no metadata available)"


# ---------------------------------------------------------------------------
# Gemini call
# ---------------------------------------------------------------------------

def _generate(context: str, n: int = N_PROMPTS) -> list[str]:
    prompt = (
        f"{context}\n\n"
        f"Generate exactly {n} short, natural search prompts (8–12 words each) "
        "that this user would type into a short video app to find content they'd enjoy.\n\n"
        "Rules:\n"
        "- Be specific to the topics described above\n"
        "- Must be interesting and engaging"
        "- Write as a user would naturally search (lowercase, conversational)\n"
        "- Return only the prompts, one per line, no numbering or extra text"
        "- It could be question, or some interesting debate on a topic."
    )
    resp  = _gemini().models.generate_content(model="gemini-3-flash-preview", contents=prompt)
    lines = [l.strip() for l in resp.text.strip().splitlines() if l.strip()]
    return lines[:n]


# ---------------------------------------------------------------------------
# Mode handlers
# ---------------------------------------------------------------------------

def cmd_user(args: argparse.Namespace) -> None:
    client   = MilvusClient(uri=MILVUS_URI)
    key      = f"{args.id}::{args.brand_id}"
    record   = _query_user(client, key)
    profiles = _load_cache("cohort_profiles")

    cohort_ids_str = record.get("cohort_ids", "0")
    context = (
        "A user on a short video platform has these content interests:\n"
        + _cohort_context(cohort_ids_str, profiles)
    )

    prompts = _generate(context)
    _print(mode="user", source_id=args.id, brand_id=args.brand_id,
           cohort_ids=cohort_ids_str, prompts=prompts)


def cmd_video(args: argparse.Namespace) -> None:
    client = MilvusClient(uri=MILVUS_URI)
    _query_video_exists(client, args.id)

    meta    = _get_video_meta(args.id)
    context = (
        "A user is watching a video on a short video platform. "
        "The video is about:\n"
        + _video_context(meta)
    )

    prompts = _generate(context)
    _print(mode="video", source_id=args.id, prompts=prompts)


def cmd_user_video(args: argparse.Namespace) -> None:
    client   = MilvusClient(uri=MILVUS_URI)
    key      = f"{args.id}::{args.brand_id}"
    record   = _query_user(client, key)
    profiles = _load_cache("cohort_profiles")

    cohort_ids_str = record.get("cohort_ids", "0")
    _query_video_exists(client, args.video_id)
    meta = _get_video_meta(args.video_id)

    context = (
        "A user on a short video platform has these established content interests:\n"
        + _cohort_context(cohort_ids_str, profiles)
        + "\n\nThey are currently watching a video about:\n"
        + _video_context(meta)
        + "\n\nGenerate interesting prompts relevant to BOTH their interests AND this video's topic."
    )

    prompts = _generate(context)
    _print(mode="user+video", source_id=args.id, brand_id=args.brand_id,
           cohort_ids=cohort_ids_str, prompts=prompts, video_id=args.video_id)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _print(
    mode: str,
    source_id: str,
    prompts: list[str],
    brand_id: int | None = None,
    cohort_ids: str | None = None,
    video_id: str | None = None,
) -> None:
    print(f"\n{'='*60}")
    print(f"  Mode   : {mode}")
    print(f"  ID     : {source_id}")
    if brand_id is not None:
        print(f"  Brand  : {brand_id}")
    if cohort_ids is not None:
        print(f"  Cohorts: [{cohort_ids}]")
    if video_id is not None:
        print(f"  Video  : {video_id}")
    print(f"{'='*60}")
    for i, p in enumerate(prompts, 1):
        print(f"  {i}. {p}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate personalised suggested prompts")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_user = sub.add_parser("user", help="Prompts from user cohort interests")
    p_user.add_argument("id",         type=str, help="user_id")
    p_user.add_argument("--brand-id", type=int, required=True)

    p_vid = sub.add_parser("video", help="Prompts from video content")
    p_vid.add_argument("id", type=str, help="video_id")

    p_uv = sub.add_parser("user-video", help="Prompts combining user interest + video context")
    p_uv.add_argument("id",          type=str, help="user_id")
    p_uv.add_argument("--brand-id",  type=int, required=True)
    p_uv.add_argument("--video-id",  type=str, required=True)

    args = parser.parse_args()

    try:
        if args.mode == "user":
            cmd_user(args)
        elif args.mode == "video":
            cmd_video(args)
        elif args.mode == "user-video":
            cmd_user_video(args)
    except (ValueError, FileNotFoundError, EnvironmentError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


main()
