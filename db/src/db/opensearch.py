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

BATCH_SIZE = 500

def _build_client() -> AsyncOpenSearch:
    return AsyncOpenSearch(
        hosts=[settings.opensearch.url],
        http_auth=(settings.opensearch.user_name, settings.opensearch.password),
        verify_certs=settings.opensearch.os_verify,
        use_ssl=True,
        ssl_show_warn=False,
        timeout=180,
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


async def scan_video_ids(brand_id: int | None = None, page_size: int = 1000) -> list[str]:
    """
    Scans all video IDs from OpenSearch using search_after pagination.
    If brand_id is provided, filters by that brand.
    Returns a flat list of video_id strings.
    """
    client = _build_client()
    try:
        query: dict[str, Any] = {"match_all": {}} if brand_id is None else {"term": {"brand_id": brand_id}}
        video_ids: list[str] = []
        search_after: list | None = None

        while True:
            body: dict[str, Any] = {
                "query": query,
                "_source": False,
                "size": page_size,
                "sort": [{"_id": "asc"}],
            }
            if search_after:
                body["search_after"] = search_after

            resp = await client.search(index=VIDEO_INDEX, body=body)
            hits = resp["hits"]["hits"]
            if not hits:
                break

            for hit in hits:
                video_ids.append(hit["_id"])
            search_after = hits[-1]["sort"]
            logger.info(f"  Scanned {len(video_ids)} video IDs so far...")

        logger.info(f"OpenSearch scan complete: {len(video_ids)} video IDs found")
        return video_ids

    finally:
        await client.close()


async def get_videos_by_ids(video_ids: list[str]) -> list[dict[str, Any]]:
    if not video_ids:
        return []

    client = _build_client()
    try:
        videos: list[dict[str, Any]] = []
        missing: list[str] = []

        batches = [video_ids[i : i + BATCH_SIZE] for i in range(0, len(video_ids), BATCH_SIZE)]
        logger.info(f"Fetching {len(video_ids)} videos in {len(batches)} batches of up to {BATCH_SIZE}")

        for idx, batch in enumerate(batches, 1):
            response = await client.mget(
                body={"ids": batch},
                index=VIDEO_INDEX,
                _source_includes=FIELDS,
            )
            for hit in response["docs"]:
                if hit.get("found"):
                    videos.append(_extract_video(hit))
                else:
                    missing.append(hit["_id"])
            logger.debug(f"  Batch {idx}/{len(batches)} done ({len(batch)} ids)")

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
