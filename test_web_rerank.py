"""
Re-ranking variant: trains WITHOUT page context (same architecture as test_web.py)
but evaluates WITH context re-ranking applied at serving time.

Re-ranking formula:
    final_score = λ * base_score + (1-λ) * ctx_score

    base_score = dot(user_emb_512d, video_finetuned_emb_512d)  — learned preference
    ctx_score  = dot(ctx_emb_768d,  video_backbone_emb_768d)   — raw ST semantic match

Both ctx_emb and backbone video live in the same ST space so dot product is meaningful.
No context signal touches training — this isolates purely serving-time re-ranking value.

Isolated from other variants:
  - cache dir  : cache_web_rerank/
  - model file : cache_web_rerank/two_tower_web_rerank.pt
  - Milvus     : video_embeddings_web_rerank / user_embeddings_web_rerank

Prints three-way comparison at the end:
  Variant A — no context          (cache_web_no_ctx/eval_metrics.pkl)
  Variant B — context in tower    (cache_web_ctx/eval_metrics.pkl)
  Variant C — re-ranking at serve (this script)
"""
import ast
import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from db.milvus import drop_collection, upsert_video_embeddings, upsert_user_embeddings
from utils.engagement import compute_engagement_scores
from utils.embeddings import embed_videos
from utils.page_context import extract_page_context, embed_page_contexts, get_user_page_ctx_embs
from utils.cohort import cluster_videos, assign_user_cohorts, build_cohort_profiles
from utils.prompt_generator import generate_all_labels
from models.two_tower import (
    TwoTowerModel,
    EngagementDataset,
    IndexMaps,
    train,
    get_video_embeddings_finetuned,
    get_user_embedding,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR        = Path("cache_web_rerank")
CSV_FILE         = "ck_user_events_with_url.csv"
VIDEO_CSV_FILE   = "new_video_data.csv"
METADATA_FILE    = "combined_metadata.json"
MODEL_FILE       = CACHE_DIR / "two_tower_web_rerank.pt"
VIDEO_COLLECTION = "video_embeddings_web_rerank"
USER_COLLECTION  = "user_embeddings_web_rerank"
MIN_INTERACTIONS = 10
TEST_RATIO       = 0.2
RERANK_LAMBDA    = 0.7   # weight on learned base score; (1 - λ) on context score

CACHE_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _save(name: str, obj: object) -> None:
    with open(CACHE_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Cached {name}")


def _load(name: str) -> object | None:
    path = CACHE_DIR / f"{name}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Loaded {name} from cache")
        return obj
    return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_events() -> pd.DataFrame:
    cached = _load("all_events")
    if cached is not None:
        return cached  # type: ignore[return-value]
    logger.info(f"Loading events from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)
    df["user_id"]  = df["user_id"].astype(str).str.strip('"')
    df["video_id"] = df["video_id"].astype(str).str.strip('"')
    # Preserve real NaN for missing URLs
    if "url" in df.columns:
        df["url"] = df["url"].astype(str).str.strip()
        df["url"] = df["url"].replace({"nan": pd.NA, "": pd.NA, "None": pd.NA, "none": pd.NA})
    else:
        df["url"] = pd.NA
    n_with_url = df["url"].notna().sum()
    logger.info(f"  {len(df)} rows, {df['user_id'].nunique()} users — "
                f"{n_with_url} rows have URL ({len(df) - n_with_url} without)")
    _save("all_events", df)
    return df


def get_videos(video_ids: list[str]) -> list[dict]:
    cached = _load("videos")
    if cached is not None:
        return cached  # type: ignore[return-value]
    logger.info(f"Loading videos from {VIDEO_CSV_FILE}...")
    df = pd.read_csv(VIDEO_CSV_FILE)
    df = df[df["video_id"].isin(set(video_ids))]

    def _parse_keywords(val: object) -> list[str]:
        if pd.isna(val) or not val:  # type: ignore[arg-type]
            return []
        try:
            parsed = ast.literal_eval(str(val))
            if isinstance(parsed, list):
                return [str(k) for k in parsed]
        except (ValueError, SyntaxError):
            pass
        return [s.strip() for s in str(val).split(",") if s.strip()]

    def _str(val) -> str:
        if pd.isna(val) if not isinstance(val, (list, dict)) else False:
            return ""
        s = str(val).strip()
        return "" if s.lower() == "nan" else s

    videos: list[dict] = []
    for _, row in df.iterrows():
        videos.append({
            "video_id":              str(row["video_id"]),
            "transcript":            _str(row.get("transcript")),
            "description_text":      _str(row.get("description_text")),
            "video_gen_description": _str(row.get("video_gen_description")),
            "keywords":              _parse_keywords(row.get("keywords")),
            "embedding_confidence":  _str(row.get("embedding_confidence")) or "none",
        })

    logger.info(f"Loaded {len(videos)} videos from CSV (of {len(video_ids)} requested)")
    _save("videos", videos)
    return videos


def get_video_embeddings_cached(videos: list[dict]) -> dict:
    cached = _load("video_embeddings")
    if cached is not None:
        return cached  # type: ignore[return-value]
    embs = embed_videos(videos)
    _save("video_embeddings", embs)
    return embs


def load_page_ctx_embs(events_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """
    Load page context embeddings per (user, brand).
    Used only in evaluation — the model is not trained with these.
    Users without URLs get a zero vector (excluded from re-rank eval).
    """
    cached = _load("user_page_ctx_embs")
    if cached is not None:
        return cached  # type: ignore[return-value]

    logger.info(f"Loading page metadata from {METADATA_FILE}...")
    with open(METADATA_FILE) as f:
        raw_metadata: dict = json.load(f)

    unique_urls = events_df["url"].dropna().astype(str).unique().tolist()
    logger.info(f"Extracting context for {len(unique_urls)} unique URLs...")
    url_context_map: dict[str, str] = {}
    missing_meta = 0
    for url in unique_urls:
        entry = raw_metadata.get(url)
        if entry:
            url_context_map[url] = extract_page_context(entry)
        else:
            url_context_map[url] = ""
            missing_meta += 1
    if missing_meta:
        logger.warning(f"{missing_meta}/{len(unique_urls)} URLs have no metadata — using empty context")

    url_emb_cached = _load("url_embeddings")
    if url_emb_cached is not None:
        url_emb_map: dict[str, np.ndarray] = url_emb_cached  # type: ignore[assignment]
    else:
        url_emb_map = embed_page_contexts(url_context_map)
        _save("url_embeddings", url_emb_map)

    user_page_ctx_embs = get_user_page_ctx_embs(events_df, url_emb_map)
    _save("user_page_ctx_embs", user_page_ctx_embs)
    return user_page_ctx_embs


def per_user_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cached_train = _load("train_df")
    cached_test  = _load("test_df")
    if cached_train is not None and cached_test is not None:
        return cached_train, cached_test  # type: ignore[return-value]

    train_rows: list[pd.DataFrame] = []
    test_rows:  list[pd.DataFrame] = []
    sparse_users = 0
    split_users  = 0

    for _, group in df.groupby("user_id"):
        group = group.sort_values("report_date")
        if len(group) < MIN_INTERACTIONS:
            train_rows.append(group)
            sparse_users += 1
        else:
            n_test = max(1, int(len(group) * TEST_RATIO))
            train_rows.append(group.iloc[:-n_test])
            test_rows.append(group.iloc[-n_test:])
            split_users += 1

    train_df = pd.concat(train_rows).reset_index(drop=True)
    test_df  = pd.concat(test_rows).reset_index(drop=True) if test_rows else pd.DataFrame()

    logger.info(
        f"Split: {split_users} users split (train/test), "
        f"{sparse_users} sparse users (train only)"
    )
    logger.info(f"  Train: {len(train_df)} rows  |  Test: {len(test_df)} rows")
    _save("train_df", train_df)
    _save("test_df", test_df)
    return train_df, test_df


# ---------------------------------------------------------------------------
# Evaluation (two passes: base + re-ranked)
# ---------------------------------------------------------------------------

def evaluate(
    model: TwoTowerModel,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    index_maps: IndexMaps,
    video_embeddings_finetuned: dict[str, np.ndarray],   # 512d — for base_score
    video_embeddings_backbone: dict[str, np.ndarray],    # 768d — for ctx_score
    user_page_ctx_embs: dict[str, np.ndarray],           # 768d per (user, brand)
    lam: float = RERANK_LAMBDA,
    k_values: list[int] = [10, 20, 50],
) -> None:
    device = next(model.parameters()).device
    model.eval()

    vid_ids = list(video_embeddings_finetuned.keys())

    # 512d finetuned matrix — for learned base scores
    finetuned_matrix = torch.tensor(
        np.stack([video_embeddings_finetuned[v] for v in vid_ids]),
        dtype=torch.float32,
    ).to(device)

    # 768d backbone matrix — for context re-ranking; L2-normalize for cosine dot
    backbone_arr = np.stack([video_embeddings_backbone[v] for v in vid_ids]).astype(np.float32)
    norms        = np.linalg.norm(backbone_arr, axis=1, keepdims=True)
    backbone_arr_norm = backbone_arr / (norms + 1e-8)

    # Engagement aggregates from training data
    train_eng     = compute_engagement_scores(train_df)
    train_ctx_map = (
        train_eng.groupby(["user_id", "brand_id"])
        .agg(
            watch_percentage=("watch_percentage", "mean"),
            views=("views", "mean"),
            likes=("likes", "mean"),
            shares=("shares", "mean"),
            comments=("comments", "mean"),
            link_clicks=("link_clicks", "mean"),
        )
        .to_dict("index")
    )

    # Metric accumulators — base (all users) and re-ranked (users with context)
    base_recalls:   dict[int, list[float]] = {k: [] for k in k_values}
    base_mrr:       list[float] = []
    base_cos:       list[float] = []

    rerank_recalls: dict[int, list[float]] = {k: [] for k in k_values}
    rerank_mrr:     list[float] = []
    rerank_cos:     list[float] = []

    with torch.no_grad():
        for (user_id, brand_id), group in test_df.groupby(["user_id", "brand_id"]):
            user_idx  = index_maps.get_user_idx(str(user_id), str(brand_id))
            brand_idx = index_maps.get_brand_idx(str(brand_id))

            ctx = train_ctx_map.get((user_id, brand_id), {})
            avg_eng = np.array([
                float(ctx.get("watch_percentage", 0.5)) / 100.0,
                min(float(ctx.get("views", 1.0)), 5.0) / 5.0,
                float(ctx.get("likes", 0.0)),
                float(ctx.get("shares", 0.0)),
                float(ctx.get("comments", 0.0)),
                float(ctx.get("link_clicks", 0.0)),
            ], dtype=np.float32)

            user_emb    = get_user_embedding(model, user_idx, brand_idx, avg_eng)
            user_tensor = torch.tensor(user_emb, dtype=torch.float32).unsqueeze(0).to(device)

            # --- Base scores (512d) ---
            base_scores = (
                (user_tensor.expand(len(vid_ids), -1) * finetuned_matrix).sum(dim=-1)
                * model.temperature
            ).cpu().numpy()

            base_ranked = [vid_ids[i] for i in np.argsort(-base_scores)]

            test_positives = set(group["video_id"].tolist()) & set(vid_ids)
            if not test_positives:
                continue

            # Base metrics
            _update_metrics(base_ranked, test_positives, base_recalls, base_mrr, k_values)
            _update_cosine(user_emb, test_positives, vid_ids, video_embeddings_finetuned, base_cos)

            # --- Re-ranked scores (λ * base + (1-λ) * ctx) ---
            ctx_emb = user_page_ctx_embs.get(f"{user_id}::{brand_id}")
            if ctx_emb is not None and np.any(ctx_emb != 0):
                ctx_norm    = ctx_emb / (np.linalg.norm(ctx_emb) + 1e-8)
                ctx_scores  = backbone_arr_norm @ ctx_norm          # (n_videos,)
                # Normalise base_scores to [0,1] so scales are comparable
                base_norm   = (base_scores - base_scores.min()) / (base_scores.max() - base_scores.min() + 1e-8)
                final_scores = lam * base_norm + (1.0 - lam) * ctx_scores
                rerank_ranked = [vid_ids[i] for i in np.argsort(-final_scores)]

                _update_metrics(rerank_ranked, test_positives, rerank_recalls, rerank_mrr, k_values)
                _update_cosine(user_emb, test_positives, vid_ids, video_embeddings_finetuned, rerank_cos)

    n_videos     = len(vid_ids)
    n_base       = len(base_mrr)
    n_rerank     = len(rerank_mrr)

    # -------------------------------------------------------------------------
    # Print results
    # -------------------------------------------------------------------------
    W = 64
    print("\n" + "=" * W)
    print(f"  VARIANT C — re-ranking at serving (λ={lam})")
    print("=" * W)

    print(f"\n  [Base — no context, {n_base} users]")
    print(f"  MRR:        {np.mean(base_mrr):.4f}")
    for k in k_values:
        print(f"  Recall@{k:<3}:  {np.mean(base_recalls[k]):.4f}")
    if base_cos:
        print(f"  Cosine gap: mean={np.mean(base_cos):.4f}  median={np.median(base_cos):.4f}")

    print(f"\n  [Re-ranked — context at serve, {n_rerank} users with URL]")
    print(f"  MRR:        {np.mean(rerank_mrr):.4f}"  if rerank_mrr else "  (no users with context)")
    for k in k_values:
        val = np.mean(rerank_recalls[k]) if rerank_recalls[k] else float("nan")
        print(f"  Recall@{k:<3}:  {val:.4f}")
    if rerank_cos:
        print(f"  Cosine gap: mean={np.mean(rerank_cos):.4f}  median={np.median(rerank_cos):.4f}")

    print(f"\n  Random baseline  ({n_videos} videos)")
    for k in k_values:
        print(f"  Recall@{k:<3}:  {k / n_videos:.4f}")

    # -------------------------------------------------------------------------
    # Three-way comparison
    # -------------------------------------------------------------------------
    var_a_path = Path("cache_web_no_ctx/eval_metrics.pkl")
    var_b_path = Path("cache_web_ctx/eval_metrics.pkl")
    if var_a_path.exists() and var_b_path.exists():
        with open(var_a_path, "rb") as f:
            var_a: dict = pickle.load(f)
        with open(var_b_path, "rb") as f:
            var_b: dict = pickle.load(f)

        print("\n" + "=" * W)
        print("  THREE-WAY COMPARISON")
        print(f"  {'Metric':<14} {'A: no-ctx':>10} {'B: ctx-tower':>13} {'C: rerank':>10}")
        print("  " + "-" * (W - 2))

        mrr_c = np.mean(rerank_mrr) if rerank_mrr else float("nan")
        print(f"  {'MRR':<14} {var_a['mrr']:>10.4f} {var_b['mrr']:>13.4f} {mrr_c:>10.4f}")
        for k in k_values:
            r_a  = var_a.get(f"recall@{k}", float("nan"))
            r_b  = var_b.get(f"recall@{k}", float("nan"))
            r_c  = np.mean(rerank_recalls[k]) if rerank_recalls[k] else float("nan")
            print(f"  {'Recall@'+str(k):<14} {r_a:>10.4f} {r_b:>13.4f} {r_c:>10.4f}")
        cg_a = var_a.get("cosine_gap_mean", float("nan"))
        cg_b = var_b.get("cosine_gap_mean", float("nan"))
        cg_c = float(np.mean(rerank_cos)) if rerank_cos else float("nan")
        print(f"  {'Cosine gap':<14} {cg_a:>10.4f} {cg_b:>13.4f} {cg_c:>10.4f}")
        print(f"\n  NOTE: C re-rank covers {n_rerank}/{n_base} users (those with URL context)")
        print("=" * W)

    # Save metrics
    metrics = {
        "variant": "web_rerank",
        "lambda":  lam,
        "base": {
            "n_eval":            n_base,
            "mrr":               float(np.mean(base_mrr)) if base_mrr else 0.0,
            **{f"recall@{k}":    float(np.mean(base_recalls[k])) for k in k_values},
            "cosine_gap_mean":   float(np.mean(base_cos)) if base_cos else 0.0,
            "cosine_gap_median": float(np.median(base_cos)) if base_cos else 0.0,
        },
        "rerank": {
            "n_eval":            n_rerank,
            "mrr":               float(np.mean(rerank_mrr)) if rerank_mrr else 0.0,
            **{f"recall@{k}":    float(np.mean(rerank_recalls[k])) if rerank_recalls[k] else 0.0
               for k in k_values},
            "cosine_gap_mean":   float(np.mean(rerank_cos)) if rerank_cos else 0.0,
            "cosine_gap_median": float(np.median(rerank_cos)) if rerank_cos else 0.0,
        },
    }
    _save("eval_metrics", metrics)
    logger.info("Saved eval metrics → cache_web_rerank/eval_metrics.pkl")


def _update_metrics(
    ranked: list[str],
    positives: set[str],
    recalls: dict[int, list[float]],
    mrr_scores: list[float],
    k_values: list[int],
) -> None:
    for k in k_values:
        recalls[k].append(len(positives & set(ranked[:k])) / len(positives))
    for rank, vid_id in enumerate(ranked, 1):
        if vid_id in positives:
            mrr_scores.append(1.0 / rank)
            return
    mrr_scores.append(0.0)


def _update_cosine(
    user_emb: np.ndarray,
    positives: set[str],
    vid_ids: list[str],
    vid_emb_map: dict[str, np.ndarray],
    cos_list: list[float],
) -> None:
    pos_sims = [
        float(np.dot(user_emb, vid_emb_map[v]))
        for v in positives
        if v in vid_emb_map
    ]
    if not pos_sims:
        return
    neg_pool = [v for v in vid_ids if v not in positives]
    if not neg_pool:
        return
    neg_ids  = np.random.choice(neg_pool, size=min(len(pos_sims), len(neg_pool)), replace=False)
    neg_sims = [float(np.dot(user_emb, vid_emb_map[v])) for v in neg_ids]
    cos_list.append(float(np.mean(pos_sims)) - float(np.mean(neg_sims)))


# ---------------------------------------------------------------------------
# User embeddings for Milvus upload
# ---------------------------------------------------------------------------

def get_all_user_embeddings(
    model: TwoTowerModel,
    train_df: pd.DataFrame,
    index_maps: IndexMaps,
) -> dict[str, np.ndarray]:
    cached = _load("user_embeddings_all")
    if cached is not None:
        return cached  # type: ignore[return-value]

    engagement_df = compute_engagement_scores(train_df)
    result: dict[str, np.ndarray] = {}

    for (user_id, brand_id), group in engagement_df.groupby(["user_id", "brand_id"]):
        uid = str(user_id)
        bid = str(brand_id)
        user_idx  = index_maps.get_user_idx(uid, bid)
        brand_idx = index_maps.get_brand_idx(bid)
        avg_eng   = np.array([
            float(group["watch_percentage"].mean()) / 100.0,
            min(float(group["views"].mean()), 5.0) / 5.0,
            float(group["likes"].mean()),
            float(group["shares"].mean()),
            float(group["comments"].mean()),
            float(group["link_clicks"].mean()) if "link_clicks" in group.columns else 0.0,
        ], dtype=np.float32)
        result[f"{uid}::{bid}"] = get_user_embedding(model, user_idx, brand_idx, avg_eng)

    _save("user_embeddings_all", result)
    logger.info(f"Computed embeddings for {len(result)} (user, brand) pairs")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("Dropping stale Milvus collections before training...")
    drop_collection(VIDEO_COLLECTION)
    drop_collection(USER_COLLECTION)

    all_df = load_events()
    train_df, test_df = per_user_split(all_df)

    all_video_ids    = all_df["video_id"].dropna().unique().tolist()
    videos           = get_videos(all_video_ids)
    video_embeddings = get_video_embeddings_cached(videos)   # 768d backbone

    # Page context embeddings — only used in evaluation, not in training
    user_page_ctx_embs = load_page_ctx_embs(all_df)

    engagement_df = compute_engagement_scores(train_df)
    index_maps    = IndexMaps(train_df)
    logger.info(f"Index maps: {index_maps.n_users} users, {index_maps.n_brands} brands")

    logger.info("Building training dataset (no page context — re-ranking is serving-time only)...")
    dataset = EngagementDataset(
        engagement_df=engagement_df,
        index_maps=index_maps,
        video_embeddings=video_embeddings,
        positive_threshold=0.05,
        negative_ratio=4,
        hard_negative_ratio=0,
    )

    logger.info("Training two-tower model (web-only, NO page context in tower)...")
    model = TwoTowerModel(
        n_users=index_maps.n_users,
        n_brands=index_maps.n_brands,
        backbone_dim=768,
        output_dim=512,
        temperature=10.0,
        use_page_ctx=False,
    )
    losses = train(model, dataset, epochs=100, batch_size=256, lr=1e-3)
    logger.info(f"Training complete. Final loss: {losses[-1]:.4f}")

    logger.info("Projecting video embeddings through finetuned video tower...")
    finetuned_videos = get_video_embeddings_finetuned(model, video_embeddings)

    evaluate(
        model=model,
        train_df=train_df,
        test_df=test_df,
        index_maps=index_maps,
        video_embeddings_finetuned=finetuned_videos,
        video_embeddings_backbone=video_embeddings,
        user_page_ctx_embs=user_page_ctx_embs,
        lam=RERANK_LAMBDA,
        k_values=[10, 20, 50],
    )

    logger.info("Computing user embeddings for all training users...")
    all_user_embs = get_all_user_embeddings(model, train_df, index_maps)

    logger.info("Clustering videos into content cohorts...")
    video_cohort_map, km = cluster_videos(finetuned_videos)
    _save("video_cohort_map", video_cohort_map)

    logger.info("Assigning users to cohorts...")
    user_cohort_map = assign_user_cohorts(engagement_df, video_cohort_map)
    _save("user_cohort_map", user_cohort_map)
    multi = sum(1 for v in user_cohort_map.values() if len(v) > 1)
    logger.info(f"  {len(user_cohort_map)} users assigned; {multi} belong to multiple cohorts")

    logger.info("Building cohort profiles...")
    cohort_profiles = build_cohort_profiles(video_cohort_map, engagement_df, videos, k=km.n_clusters)
    logger.info("Generating cohort labels via Claude...")
    cohort_profiles = generate_all_labels(cohort_profiles, video_cohort_map, videos)
    _save("cohort_profiles", cohort_profiles)

    logger.info(f"Saving model → {MODEL_FILE}")
    torch.save(model.state_dict(), MODEL_FILE)

    logger.info("Upserting embeddings to Milvus...")
    video_brand_map = all_df.groupby("video_id")["brand_id"].first().to_dict()
    upsert_video_embeddings(finetuned_videos, video_brand_map, collection=VIDEO_COLLECTION)
    upsert_user_embeddings(all_user_embs, user_cohort_map, collection=USER_COLLECTION)


main()
