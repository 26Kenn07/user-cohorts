"""
Flexible similarity queries against Milvus embeddings.

Usage:
    uv run infer.py user-to-video  <user_id>   [--top-k 20]
    uv run infer.py video-to-video <video_id>  [--top-k 20]
    uv run infer.py video-to-user  <video_id>  [--top-k 20]
    uv run infer.py user-to-user   <user_id>   [--top-k 20]
"""

import argparse
import logging
import os

from pymilvus import MilvusClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MILVUS_URI       = os.environ.get("MILVUS_URI", "http://localhost:19530")
VIDEO_COLLECTION = os.environ.get("VIDEO_COLLECTION", "video_embeddings")
USER_COLLECTION  = os.environ.get("USER_COLLECTION", "user_embeddings")


SOURCE_EXTRA_FIELDS = {
    USER_COLLECTION:  ["user_id", "brand_id", "cohort_ids"],
    VIDEO_COLLECTION: ["brand_id", "has_engagement"],
}


def get_source_record(client: MilvusClient, collection: str, id_field: str, id_value: str) -> dict:
    fields = ["embedding"] + SOURCE_EXTRA_FIELDS.get(collection, [])
    records = client.query(
        collection_name=collection,
        filter=f'{id_field} == "{id_value}"',
        output_fields=fields,
    )
    if not records:
        raise ValueError(f"'{id_value}' not found in collection '{collection}'")
    return records[0]


def search(
    client: MilvusClient,
    source_collection: str,
    target_collection: str,
    source_id_field: str,
    source_id: str,
    target_id_field: str,
    target_extra_fields: list[str],
    top_k: int,
    exclude_self: bool,
    brand_id: int | None = None,
) -> None:
    logger.info(f"Fetching embedding for {source_id_field}={source_id}...")
    source = get_source_record(client, source_collection, source_id_field, source_id)
    embedding = source["embedding"]

    filter_expr = f"brand_id == {brand_id}" if brand_id is not None else ""

    limit = top_k + 1 if exclude_self else top_k
    results = client.search(
        collection_name=target_collection,
        data=[embedding],
        anns_field="embedding",
        limit=limit,
        output_fields=[target_id_field] + target_extra_fields,
        filter=filter_expr,
        search_params={"metric_type": "IP", "params": {"nprobe": 128}},
    )

    hits = results[0]
    if exclude_self:
        hits = [h for h in hits if h["entity"][target_id_field] != source_id]
    hits = hits[:top_k]

    # Print source metadata
    src_meta = {k: v for k, v in source.items() if k != "embedding"}
    src_meta_str = "  ".join(f"{k}={v}" for k, v in src_meta.items())

    print(f"\n{'='*60}")
    print(f"  Query  : {source_id_field} = {source_id}")
    print(f"  Source : {src_meta_str}")
    print(f"  Mode   : {source_collection} → {target_collection}")
    if brand_id is not None:
        print(f"  Filter : brand_id = {brand_id}")
    print(f"{'='*60}")
    for i, hit in enumerate(hits, 1):
        entity = hit["entity"]
        score  = hit["distance"]
        extra  = "  ".join(f"{f}={entity[f]}" for f in target_extra_fields)
        print(f"  {i:2}. {entity[target_id_field]:<40}  score={score:.4f}  {extra}")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Milvus similarity queries")
    parser.add_argument(
        "mode",
        choices=["user-to-video", "video-to-video", "video-to-user", "user-to-user"],
        help="Query direction",
    )
    parser.add_argument("id", type=str, help="Source user_id or video_id")
    parser.add_argument("--top-k",    type=int, default=20)
    parser.add_argument("--brand-id", type=int, default=None, help="Filter results to a specific brand")
    args = parser.parse_args()

    client = MilvusClient(uri=MILVUS_URI)

    MODE_CONFIG = {
        #                 src_coll        tgt_coll        src_field         tgt_field         tgt_extra                              same_coll
        "user-to-video":  (USER_COLLECTION,  VIDEO_COLLECTION, "user_brand_key", "video_id",       ["brand_id", "has_engagement"],        False),
        "video-to-video": (VIDEO_COLLECTION, VIDEO_COLLECTION, "video_id",       "video_id",       ["brand_id", "has_engagement"],        True),
        "video-to-user":  (VIDEO_COLLECTION, USER_COLLECTION,  "video_id",       "user_brand_key", ["user_id", "cohort_ids", "brand_id"],   False),
        "user-to-user":   (USER_COLLECTION,  USER_COLLECTION,  "user_brand_key", "user_brand_key", ["user_id", "cohort_ids", "brand_id"],   True),
    }

    src_coll, tgt_coll, src_field, tgt_field, tgt_extra, exclude_self = MODE_CONFIG[args.mode]

    # For user-source modes, construct composite key from user_id + brand_id
    source_id = args.id
    if src_field == "user_brand_key":
        if args.brand_id is None:
            parser.error(f"--brand-id is required for {args.mode} mode")
        source_id = f"{args.id}::{args.brand_id}"

    search(
        client=client,
        source_collection=src_coll,
        target_collection=tgt_coll,
        source_id_field=src_field,
        source_id=source_id,
        target_id_field=tgt_field,
        target_extra_fields=tgt_extra,
        top_k=args.top_k,
        exclude_self=exclude_self,
        brand_id=args.brand_id,
    )


main()
