# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Codex will review your output once you are done.

## Commands

```bash
# Full retrain pipeline
uv run test.py

# Reset before retrain — two modes:
uv run reset.py                # full reset: clears all cache + Milvus embeddings
uv run reset.py --keep-data    # keeps CSV/embedding caches, clears only model outputs
                                # use --keep-data when only hyperparameters changed
                                # use full reset when new CSV data is available

# Fetch fresh event data from ClickHouse (run before full reset on retrain cycles)
uv run fetch_data.py

# Ingest missing videos for a brand into Milvus (run after training)
uv run ingest_missing_videos.py --brand-id 2357          # Ted
uv run ingest_missing_videos.py --brand-id 2357 --dry-run # preview count only

# Similarity queries with rich context
uv run infer_v2.py user-to-video <user_id> --brand-id <id> [--top-k 20]
uv run infer_v2.py video-to-video <video_id>
uv run infer_v2.py video-to-user <video_id>
uv run infer_v2.py user-to-user <user_id> --brand-id <id>
uv run infer_v2.py list-cohorts

# Add a dependency to a workspace member
uv add <package> --package db        # or utils, models
uv add <package>                     # root package
```

## Architecture

This is a **uv workspace** with four packages: `db`, `utils`, `models`, and a root package. All run from the repo root via `uv run`.

### Data Sources
- **ClickHouse** (`db/src/db/clickhouse.py`) — user event logs (`genuin_events_logs_001`). Returns one row per `(user_id, video_id, brand_id, report_date)` with aggregated `views`, `likes`, `shares`, `comments`, `link_clicks`, `watch_percentage`. Applies a `qualified_users` CTE filter: only users with `> min_events` (default 10) event rows per brand are included. `fetch_data.py` pages through results in batches of 50K and saves to `new_ck_user_events.csv`.
- **OpenSearch** (`db/src/db/opensearch.py`) — video metadata from index `genuin_loop_video_index`. Fields: `transcript`, `description_text`, `video_gen_description`, `keywords.explicit[].keyword`. Fetched via `mget` for specific video IDs, or via `scan_video_ids()` for all videos of a brand.

### Two-Tower Model (`models/src/models/two_tower.py`)

**VideoTower** — takes a **768d** sentence-transformer embedding (precomputed, frozen backbone) and projects through a trainable MLP to **512d**.

**UserTower** — YouTube DNN-style. Takes:
- `user_idx`: learnable per-(user, brand) embedding (`padding_idx=0` for cold-start)
- `brand_idx`: learnable per-brand embedding
- `engagement`: 6 raw features — `watch_ratio, views_capped, likes, shares, comments, link_clicks`

`IndexMaps` maps string `user_id`/`brand_id` pairs to integer indices. A user active on 3 brands gets 3 embedding slots. Users not in training get `user_idx=0` — the model falls back to brand + engagement signals only (cold-start path).

**Training** — InfoNCE loss (temperature=0.07): each positive paired with `negative_ratio=4` random unwatched videos; cross-entropy over 5-way classification. `hard_negative_ratio` controls how many of the 4 negatives are semantically similar (backbone cosine) vs random — keep at 0 (all random), hard negatives based on backbone similarity hurt recommendation quality. AdamW + CosineAnnealingLR + gradient clipping. Best config: `positive_threshold=0.05`, `epochs=100`, `batch_size=256`.

### Pipeline Flow (`test.py`)

```
new_ck_user_events.csv  →  load_events()  →  per_user_split (80/20 chronological)
new_video_data.csv      →  get_videos()   →  embed_videos (ST backbone, 768d)
                                                      ↓
                                          compute_engagement_scores()
                                                      ↓
                                          EngagementDataset (InfoNCE samples)
                                                      ↓
                                          TwoTowerModel.train() → cache/two_tower.pt
                                                      ↓
                                          get_video_embeddings_finetuned() (512d)
                                                      ↓
                                          evaluate() → MRR, Recall@10/20/50, cosine gap
                                                      ↓
                                          cluster_videos() → K-Means cohorts
                                          assign_user_cohorts() → per-user cohort list
                                          generate_all_labels() → Gemini cohort labels
                                                      ↓
                                          upsert to Milvus (video + user embeddings)
```

All expensive steps are cached to `cache/*.pkl`. `reset.py` manages cache invalidation cleanly — prefer it over manual `rm`.

**Retrain cadence**: every 3-4 days. Run `fetch_data.py` → `reset.py` → `test.py`. Use `reset.py --keep-data` only when tuning hyperparameters without new data.

### Engagement Scoring (`utils/src/utils/engagement.py`)

Score per `(user, video)` = `(watch_percentage/100)×1.0 + views.clip(3)×0.5 + likes×3.0 + shares×5.0 + comments×3.0 + link_clicks×4.0`, then **normalized per brand** (divided by the brand's max score). Per-brand normalization is critical — global normalization collapses scores when one brand has a single super-engaged user.

### Video Embedding (`utils/src/utils/embeddings.py`)

Backbone: **`all-mpnet-base-v2`** (768d, 110M params). Weighted combination per video: transcript (0.4) + description_text (0.3) + video_gen_description (0.25) + mean(keyword embeddings) (0.05). Missing fields skipped gracefully. Batches all text into a single GPU encode pass per field.

### Cohort Clustering (`utils/src/utils/cohort.py`)

K-Means on finetuned **512d video embeddings** (not user embeddings). Optimal k via silhouette score (k=3–8). Each user is assigned to one or more cohorts based on which content clusters account for ≥15% of their total engagement score — users with diverse interests get multiple cohorts. Cohort labels generated by Gemini 2.5 Flash (`generate_all_labels()`) based on actual video descriptions, not just keywords.

### Milvus (`db/src/db/milvus.py`)

Two collections at `MILVUS_URI` (from `.env`):
- `video_embeddings` — 512d finetuned embeddings, fields: `video_id`, `brand_id`, `has_engagement`, `updated_at`. Filtered by `brand_id` at query time. `has_engagement=False` marks cold-start videos (content-only, not in training events).
- `user_embeddings` — 512d embeddings, fields: `user_brand_key` (`user_id::brand_id`), `user_id`, `brand_id`, `cohort_ids` (comma-separated), `updated_at`.

**Cold-start users** (<10 events, not in training) do **not** get a pre-stored Milvus embedding — their embedding is computed on-the-fly at inference via `get_user_embedding()` using `user_idx=0` + current engagement features. Do not store these in Milvus; the embedding changes significantly with each new event.

After training, run `ingest_missing_videos.py --brand-id <id>` to backfill any brand videos that exist in OpenSearch but not in Milvus (typically <1% gap).

### Page Context (`utils/src/utils/page_context.py`)

Extracts keyword strings from URL metadata (title, description, tags). Uses KeyBERT backed by the shared `all-mpnet-base-v2` model for keyword extraction, or reads pre-extracted `bert_keywords` if available. `get_user_page_ctx_embs()` averages context embeddings across all URLs a (user, brand) pair was seen on. Only JS-sourced events carry page context — treat as an optional user feature with `unknown` category fallback for missing values.

### Multi-Brand Design

- One shared model across all brands — per-brand models are not used (data too sparse per brand).
- `brand_id` is a learnable embedding in the user tower, so brand-specific behavior is learned implicitly.
- User identity: `context_device_advertising_id` (device-level global ID, cross-brand) vs `identity_id` (brand-local). The `identity_type` column captures which was used. Cross-brand signal via `context_ad_id` is the intended mechanism for cold-start on a new brand.
- Currently active brand IDs in training data: `(1729, 2023, 2075, 2556, 2558, 2314, 2357, 2476, 2557, 2701, 2764, 2790, 2793, 2801, 2808, 3099)`. Brand 2357 = Ted.

### Known Issues / Watch-outs

- `embed_users()` in `embeddings.py` has a stale default `dim=384` — it's unused in the main pipeline (user embeddings come from the UserTower, not this function) but would need fixing if called directly.
- `str(float('nan'))` produces the literal string `"nan"` in Python (NaN is truthy, so `nan or ""` returns `nan`). Video metadata loaded from CSV must use `pd.isna()` checks before `str()` conversion — see `_str()` helper in `test.py:get_videos()`. `infer_v2.py:_video_snippet()` also filters `"nan"` strings defensively.
- Hard negatives based on backbone embedding similarity hurt recommendation quality (tested: cosine gap 0.60 → 0.31). Keep `hard_negative_ratio=0`.
