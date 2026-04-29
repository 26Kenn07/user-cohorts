# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Codex will review your output once you are done.

## Commands

```bash
# Run the main pipeline
uv run test.py

# Add a dependency to a workspace member
uv add <package> --package db        # or utils, models
uv add <package>                     # root package

# Clear cached data (forces re-fetch from ClickHouse + OpenSearch)
rm -rf cache/

# Clear only embeddings cache (forces re-embedding with sentence-transformer)
rm cache/video_embeddings.pkl

# Clear only DB caches (forces re-fetch, keeps embeddings)
rm cache/train_df.pkl cache/test_df.pkl cache/videos.pkl
```

## Architecture

This is a **uv workspace** with four packages: `db`, `utils`, `models`, and a root package. All run from the repo root via `uv run`.

### Data Sources
- **ClickHouse** (`db/src/db/clickhouse.py`) — user event logs (`genuin_events_logs_001`). Events: `video_impression`, `video_watched`, `video_sparked` (like), `video_shared`, `commented_on_video`. Returns one row per `(user_id, video_id, report_date)` with aggregated `views`, `likes`, `shares`, `comments`, `watch_percentage`.
- **OpenSearch** (`db/src/db/opensearch.py`) — video metadata from index `genuin_loop_video_index`. Fields used: `transcript`, `description_text`, `video_gen_description`, `keywords.explicit[].keyword`. Fetched via `mget` for only the video_ids present in the event data.

### Two-Tower Model (`models/src/models/two_tower.py`)

**VideoTower** — takes a 384d sentence-transformer embedding (precomputed, frozen backbone) and projects through a trainable MLP to 128d.

**UserTower** — YouTube DNN-style. Takes:
- `user_idx`: learnable embedding per user (`padding_idx=0` for unknown/new users)
- `brand_idx`: learnable embedding per brand
- `engagement`: 5 raw features (watch_ratio, views_capped, likes, shares, comments)

New/cold-start users get `user_idx=0` — the model falls back to brand context + engagement features. This means evaluation covers ALL users, not just warm ones.

**Training** — BCEWithLogitsLoss on positive/negative pairs. Positives = interactions with `score >= threshold`. Negatives = random unwatched videos (4:1 ratio). Temperature scaling (`default=10.0`) sharpens score distribution. AdamW + CosineAnnealingLR + gradient clipping.

**IndexMaps** — maps string `user_id`/`brand_id` to integer indices. Built from training data only; test users not in training automatically get `user_idx=0`.

### Pipeline Flow (`test.py`)

```
ClickHouse (train_df)  →  compute_engagement_scores  →  EngagementDataset
OpenSearch (videos)    →  embed_videos (ST backbone)  →  VideoTower MLP
                                                           ↓
                                                     TwoTowerModel.train()
                                                           ↓
                                              get_video_embeddings_finetuned()
                                                           ↓
                                                       evaluate()
```

All expensive steps are cached to `cache/*.pkl`. Delete selectively to re-run specific steps.

### Engagement Scoring (`utils/src/utils/engagement.py`)

Score per `(user, video)` = `(watch_percentage/100)*1.0 + views_capped*0.5 + likes*3.0 + shares*5.0 + comments*3.0`, normalized to [0,1] per brand.

### Video Embedding (`utils/src/utils/embeddings.py`)

Weighted combination: transcript (0.4) + description (0.3) + AI description (0.2) + mean(keyword embeddings) (0.1). Missing fields are skipped gracefully. Uses `sentence-transformers/all-MiniLM-L6-v2` on MPS/CUDA/CPU auto-detected.

### Cohort Clustering (`utils/src/utils/cohort.py`)

K-Means on finetuned 128d user embeddings. Optimal k found via silhouette score (k=3–8). Cohort profiles built from top engagement-weighted keywords across member videos.

### Multi-Brand Design

- `brand_id` is an embedding feature in the user tower — the model learns brand-specific behavior
- `context_ad_id` (device-level global ID) vs `identity_id` (brand-local) are captured in `identity_type` column. Cross-brand signal via `context_ad_id` is the intended mechanism for cold-start on a new brand.
- One shared model across all brands — per-brand models are not used (data too sparse per brand).

### Milvus (planned)
- One collection for video embeddings (128d, finetuned), filtered by `brand_id` at query time.
- User embeddings computed at request time via `get_user_embedding()` — not stored permanently since they update with each interaction.
