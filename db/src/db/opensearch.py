import logging
from typing import Any

from opensearchpy import AsyncOpenSearch
from utils.config import settings  # pyright: ignore[reportMissingTypeStubs]

logger = logging.getLogger(__name__)

VIDEO_INDEX = "genuin_loop_video_index"

FIELDS = [
    "transcript",
    "description_text",
    "video_gen_description",
    "keywords",
]


def _build_client() -> AsyncOpenSearch:
    return AsyncOpenSearch(
        hosts=[settings.opensearch.url],
        http_auth=(settings.opensearch.user_name, settings.opensearch.password),
        verify_certs=settings.opensearch.os_verify,
        ssl_show_warn=False,
    )


def _parse_keywords(raw_keywords: Any) -> list[str]:
    """
    Extracts keyword strings from keywords.explicit list.
    Gracefully handles missing, malformed, or unexpected structures.
    """
    if not raw_keywords or not isinstance(raw_keywords, dict):
        return []

    explicit = raw_keywords.get("explicit")
    if not explicit or not isinstance(explicit, list):
        return []

    keywords = []
    for item in explicit:
        if not isinstance(item, dict):
            continue
        keyword = item.get("keyword")
        if keyword and isinstance(keyword, str) and keyword.strip():
            keywords.append(keyword.strip())

    return keywords


def _confidence(source: dict[str, Any]) -> str:
    filled = sum(1 for f in ["description_text", "video_gen_description", "transcript", "keywords"] if source.get(f))
    if filled >= 3:
        return "high"
    if filled == 2:
        return "medium"
    if filled == 1:
        return "low"
    return "none"


def _extract_video(hit: dict[str, Any]) -> dict[str, Any]:
    source = hit.get("_source") or {}
    return {
        "video_id": hit["_id"],
        "transcript": source.get("transcript") or "",
        "description_text": source.get("description_text") or "",
        "video_gen_description": source.get("video_gen_description") or "",
        "keywords": _parse_keywords(source.get("keywords")),  # list[str], equal weight
        "embedding_confidence": _confidence(source),
    }


async def get_videos_by_ids(video_ids: list[str]) -> list[dict[str, Any]]:
    if not video_ids:
        return []

    client = _build_client()
    try:
        response = await client.mget(
            body={"ids": video_ids},
            index=VIDEO_INDEX,
            _source_includes=FIELDS,
        )

        videos = []
        missing = []
        for hit in response["docs"]:
            if hit.get("found"):
                videos.append(_extract_video(hit))
            else:
                missing.append(hit["_id"])

        if missing:
            logger.warning(
                f"{len(missing)} video_ids not found in OpenSearch: "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )

        logger.info(f"Fetched {len(videos)}/{len(video_ids)} videos from OpenSearch")
        return videos

    except Exception as e:
        logger.error(f"Failed to fetch videos from OpenSearch: {e}")
        raise e

    finally:
        await client.close()
