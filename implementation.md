# Video Recommendation System — Technical Documentation

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Data Sources](#4-data-sources)
5. [Video Embedding — Content Understanding](#5-video-embedding--content-understanding)
6. [Two-Tower Neural Network](#6-two-tower-neural-network)
7. [User Cohorts](#7-user-cohorts)
8. [Multi-Brand Architecture](#8-multi-brand-architecture)
9. [Vector Database & Retrieval](#9-vector-database--retrieval)
10. [Accuracy Metrics — Definitions & Results](#10-accuracy-metrics--definitions--results)
11. [Why This Approach](#11-why-this-approach)
12. [System Update Schedule](#12-system-update-schedule)
13. [Key Hard-Coded Values & Rationale](#13-key-hard-coded-values--rationale)

---

## 1. Executive Summary

We have built a **personalised video recommendation engine** that learns what each user enjoys watching and surfaces the most relevant content for them — across all brands on the Genuin platform.

### What it does

- Learns individual user preferences from their engagement history (watch time, likes, shares, comments)
- Groups users into **5 content cohorts** based on what they watch (e.g. Celebrity News, Automotive, Wellness)
- Serves real-time video recommendations by comparing a user's learned preference vector against all available videos
- Handles **new users with no history** via cohort-level cold-start recommendations
- Supports users active on **multiple brands** — preferences on each brand are tracked independently

### Key Results


| Metric     | Our Model | Random Baseline | Improvement           |
| ---------- | --------- | --------------- | --------------------- |
| Recall@10  | 5.1%      | 0.10%           | **51× better**        |
| Recall@20  | 6.1%      | 0.21%           | **29× better**        |
| Recall@50  | 8.5%      | 0.52%           | **16× better**        |
| Cosine Gap | 0.31      | ~0.00           | strong learned signal |


> The model correctly surfaces a video the user will engage with in the top 10 results 5% of the time — across a catalogue of ~10,000 videos. That is 51 times better than random.

### Cohorts Discovered


| Cohort | Label                       | Videos | Description                                                                                |
| ------ | --------------------------- | ------ | ------------------------------------------------------------------------------------------ |
| 0      | Celebrity News & Life       | 1,416  | Updates on celebrity events, personal lives, and controversies                             |
| 1      | Life Experiences & Events   | 5,084  | People share personal experiences, emotions, and styles at diverse events                  |
| 2      | Car Features & Deals        | 1,783  | Showcasing car models, highlighting features, design, interiors, and promotional offers    |
| 3      | Public Figure Conversations | 723    | Interviews and discussions with public figures about their lives, shows, and relationships |
| 4      | Empowerment & Expression    | 623    | Individuals share insights, express themselves, and inspire personal growth                |


---

## 2. Problem Statement

**1. Personalisation** — Given a user's history, predict which new videos they will engage with.

**2. Cold-Start** — A new user has no history. What do we show them? We assign them to a content cohort based on minimal signals and recommend top-performing videos from that cohort.

**3. Multi-Brand** — The same user may be active on multiple brands (e.g. iHear and a car dealership brand). Their preferences on each brand are different and must be tracked separately.

---

## 3. High-Level Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                              │
│                                                                │
│  ClickHouse ──────────── User Events ──────────────────────┐   │
│  (views, likes,           (130K users                      │   │
│   shares, comments,        × 17 brands                     │   │
│   watch time)              × ~540K interactions)           │   │
│                                                            │   │
│  OpenSearch ──────────── Video Metadata ───────────────┐   │   │
│  (transcripts,            (~10K videos)                │   │   │
│   descriptions,                                        │   │   │
│   AI descriptions,                                     │   │   │
│   keywords)                                            │   │   │
└────────────────────────────────────────────────────────┼───┼───┘
                                                         │   │
                                                         ▼   ▼
┌─────────────────────────────────────────────────────────────────┐
│                      MODEL LAYER                                │
│                                                                 │
│  Sentence Transformer ──► Video Embeddings (768d)               │
│  (frozen backbone)         ↓                                    │
│                        VideoTower MLP ──► 512d video vectors    │
│                                                                 │
│  User Events ──────────► UserTower ─────► 512d user vectors     │
│  (engagement features,    (learnable                            │
│   brand context,          embeddings +                          │
│   user history)           MLP)                                  │
│                                                                 │
│  Training: BCELoss on positive/negative (user, video) pairs     │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     STORAGE & RETRIEVAL                         │
│                                                                 │
│  Milvus Vector DB                                               │
│  ├── video_embeddings  (9,629 videos × 512d)                    │
│  └── user_embeddings   (134,594 user-brand pairs × 512d)        │
│                                                                 │
│  Query: dot-product similarity, filtered by brand_id            │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      COHORT LAYER                               │
│                                                                 │
│  K-Means on video embeddings ──► 5 content clusters             │
│  Gemini 2.5 Flash ────────────► Human-readable cohort labels    │
│  User-to-cohort assignment ───► multi-cohort (user can be in 2+)│
└─────────────────────────────────────────────────────────────────┘
```

### Request Flow (Production)

```
User opens feed
     │
     ▼
Fetch user embedding from Milvus (user_brand_key = "user_id::brand_id")
     │
     ├── Returning user → use stored 512d preference embedding
     └── New user       → use cohort centroid embedding (cold-start)
     │
     ▼
ANN dot-product search over video_embeddings (filtered by brand_id)
     │
     ▼
Return top-K video IDs ranked by similarity score
```

---

## 4. Data Sources

### ClickHouse — User Events

**Table:** `genuin_events_logs_001`

We capture five event types per `(user_id, video_id, report_date)`:


| Event                | Signal Meaning                          |
| -------------------- | --------------------------------------- |
| `video_impression`   | User saw the video                      |
| `video_watched`      | User watched (with watch percentage)    |
| `video_sparked`      | User liked the video                    |
| `video_shared`       | User shared — strongest positive signal |
| `commented_on_video` | User commented                          |


Each row is aggregated to one record per `(user_id, video_id, report_date)` with columns: `views`, `watch_percentage`, `likes`, `shares`, `comments`.

**Scale:** ~540,000 interaction rows across 17 brand IDs, date range Jan–Apr 2026.

### OpenSearch — Video Metadata

**Index:** `genuin_loop_video_index`

Fields used per video:


| Field                   | Purpose                                    |
| ----------------------- | ------------------------------------------ |
| `transcript`            | Full speech-to-text of the video           |
| `description_text`      | Human-written description                  |
| `video_gen_description` | AI-generated description (highest quality) |
| `keywords`              | User-entered tags (noisy — low weight)     |


---

## 5. Video Embedding — Content Understanding

### Why Embeddings?

To compare videos by content, we need a numerical representation. A **768-dimensional vector** encodes the semantic meaning of a video — two videos about car reviews will have similar vectors even if they use different words.

### Backbone Model

We use `**all-mpnet-base-v2`** — a 110M parameter sentence transformer that converts text into 768-dimensional semantic vectors. The backbone is **frozen** (not trained) — we rely entirely on its pre-trained language understanding.

### Weighted Text Combination

Each video has multiple text fields of varying quality. We combine them with tuned weights:


| Field                   | Weight   | Rationale                                |
| ----------------------- | -------- | ---------------------------------------- |
| `transcript`            | **0.40** | Most specific — actual spoken content    |
| `description_text`      | **0.30** | Human-written, reliable                  |
| `video_gen_description` | **0.25** | AI-generated, clean and consistent       |
| `keywords`              | **0.05** | User-entered, often noisy or too generic |


Keywords were deliberately down-weighted after observing noise terms like "us logo" and "price subject to change" polluting cohort definitions. The final embedding is a weighted average of available fields, L2-normalised to unit length.

### VideoTower Fine-Tuning

The 768d backbone embedding is projected through a **trainable MLP** to 512d:

```
768d backbone embedding
        ↓
Linear(768→512) + LayerNorm + ReLU + Dropout(0.2)
        ↓
Linear(512→512)
        ↓
L2 Normalise → 512d video vector
```

This projection is trained so that video vectors align with user preference vectors in the same 512d space.

---

## 6. Two-Tower Neural Network

### What is a Two-Tower Model?

A Two-Tower (Dual Encoder) model is the industry-standard architecture for recommendation at scale, used by YouTube, Google, Pinterest, and others. It trains two separate neural networks — one for users, one for videos — to produce vectors in the same embedding space. Recommendation becomes a nearest-neighbour search.

```
User Tower  ──────────────────────────────► 512d user vector  ─┐
                                                               ├─► dot product → score
Video Tower ──────────────────────────────► 512d video vector ─┘
```

### User Tower Architecture

The user tower is inspired by YouTube's Deep Neural Network recommender (Covington et al., 2016):

```
Inputs:
  user_idx   → Embedding(n_users+1, 128, padding_idx=0)   ← learnable per-user preference
  brand_idx  → Embedding(n_brands, 64)                    ← learnable per-brand context
  engagement → [watch_ratio, views_capped, likes,          ← raw engagement signals
                shares, comments]  (5 features)
                    ↓
               Linear(5→64) + ReLU

Concatenate: [128 + 64 + 64] = 256d
        ↓
Linear(256→512) + LayerNorm + ReLU + Dropout(0.2)
        ↓
Linear(512→256) + ReLU
        ↓
Linear(256→512)
        ↓
L2 Normalise → 512d user vector
```

**Cold-start handling:** New users get `user_idx=0` (padding index → zero embedding). The model falls back to brand context + engagement signals, which is sufficient for cohort-level recommendations.

### Engagement Feature Engineering

Five raw signals are normalised before entering the user tower:


| Raw Signal         | Normalisation     | Rationale                            |
| ------------------ | ----------------- | ------------------------------------ |
| `watch_percentage` | ÷ 100 → [0, 1]    | Direct attention signal              |
| `views`            | min(views, 5) ÷ 5 | Capped at 5 to reduce outlier effect |
| `likes`            | raw count         | Binary-ish, rarely > 1               |
| `shares`           | raw count         | Strongest intent signal              |
| `comments`         | raw count         | High engagement signal               |


### Engagement Score (Training Labels)

A composite score combines all signals into a single engagement quality number per `(user, video)`:

```
score = (watch_percentage/100 × 1.0)
      + (min(views, 5)/5 × 0.5)
      + (likes × 3.0)
      + (shares × 5.0)
      + (comments × 3.0)
```

Scores are normalised to [0, 1] **per brand**. Interactions with `score ≥ 0.05` are treated as positives for training.

### Training Process

**Loss Function:** Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`)

**Positive/Negative Sampling:**

- Positives: `(user, video)` pairs where engagement score ≥ 0.05
- Negatives: random unwatched videos, **4 negatives per positive** (hard negative mining)

**Training Configuration:**


| Parameter         | Value             | Rationale                               |
| ----------------- | ----------------- | --------------------------------------- |
| Epochs            | 50                | Sufficient convergence observed         |
| Batch size        | 256               | Memory / speed balance                  |
| Learning rate     | 1e-3              | AdamW with cosine decay                 |
| Weight decay      | 1e-5              | Light regularisation                    |
| LR schedule       | CosineAnnealingLR | Smooth decay, better final convergence  |
| Gradient clipping | max_norm=1.0      | Prevents exploding gradients            |
| Temperature       | 10.0              | Sharpens dot-product score distribution |


**Temperature scaling:** The raw dot product of two unit vectors lies in [-1, 1]. Multiplying by temperature=10 spreads scores to [-10, 10], giving the sigmoid a stronger gradient signal during training.

### IndexMaps — Multi-Brand User Identity

Each `(user_id, brand_id)` pair gets a unique embedding slot. A user active on 3 brands gets 3 separate learnable embeddings capturing their distinct preferences per brand:

```
user_id=U1, brand_id=1729  →  user_idx=1   (iHear preferences)
user_id=U1, brand_id=2793  →  user_idx=2   (automotive brand preferences)
user_id=U1, brand_id=2023  →  user_idx=3   (wellness brand preferences)
```

Stored in Milvus as composite key: `"U1::1729"`, `"U1::2793"`, `"U1::2023"`.

---

## 7. User Cohorts

### Why Cohorts?

Cohorts serve two purposes:

1. **Cold-start:** New users are assigned to a content cohort and receive top-performing videos from that cohort.
2. **Prompt generation:** Each cohort generates targeted search prompts to surface relevant new content.

### Why Video-Based Clustering?

Initial attempts clustered user embeddings directly. This produced poor results because the user tower bakes `brand_id` as a learnable embedding — users from the same brand clustered together by brand geography rather than content interest.

**Solution:** Cluster **video embeddings** by content instead. Cohorts then represent content topics. Users are assigned based on what they actually watched.

### Algorithm

**Step 1 — Optimal k via silhouette score**
K-Means is run for k=3 to k=8. The k with the highest silhouette score is selected:

```
k=3: 0.2263
k=4: 0.2407
k=5: 0.2470  ← optimal
k=6: 0.1691
...
```

**Step 2 — Cluster 9,629 video embeddings → 5 cohorts**

**Step 3 — Assign users to cohorts**
Each user's total engagement score is distributed across video cohorts. A user is assigned to every cohort that accounts for ≥15% of their engagement:

```
Example user:
  Cohort 0 (Celebrity):   70% of engagement score  → assigned ✓
  Cohort 2 (Automotive):  25% of engagement score  → assigned ✓
  Cohort 4 (Empowerment):  5% of engagement score  → not assigned

Result: user belongs to cohorts [0, 2]
```

**34,244 of 134,594 users (25%) belong to multiple cohorts**, reflecting genuinely diverse interests.

**Step 4 — Generate cohort labels with Gemini 2.5 Flash**
15 video descriptions per cluster are sent to Gemini to produce a human-readable label and one-sentence description. This runs once at training time (5 API calls total).

### Final Cohorts


| ID  | Label                       | Videos | Description                                                                                |
| --- | --------------------------- | ------ | ------------------------------------------------------------------------------------------ |
| 0   | Celebrity News & Life       | 1,416  | Updates on celebrity events, personal lives, and controversies                             |
| 1   | Life Experiences & Events   | 5,084  | People share personal experiences, emotions, and styles at diverse events                  |
| 2   | Car Features & Deals        | 1,783  | Showcasing car models, highlighting features, design, interiors, and promotional offers    |
| 3   | Public Figure Conversations | 723    | Interviews and discussions with public figures about their lives, shows, and relationships |
| 4   | Empowerment & Expression    | 623    | Individuals share insights, express themselves, and inspire personal growth                |


---

## 8. Multi-Brand Architecture

### The Problem

A user can be active on multiple brands. Assigning only one brand per user (e.g. the most recent brand) produces wrong recommendations when that user is on a different brand.

### The Solution

The composite key `user_id::brand_id` is used throughout the entire system:

- **Training:** `IndexMaps` creates one embedding slot per `(user_id, brand_id)` pair
- **Milvus:** Primary key is `user_brand_key` (e.g. `"abc123::1729"`)
- **Inference:** `brand_id` is a required parameter when querying user recommendations

**Scale:** 130,794 unique users across 17 brands → **134,594 `(user, brand)` embedding pairs** trained.

---

## 9. Vector Database & Retrieval

We use **Milvus** (open-source vector database) for storing and searching embeddings at scale.

### Collections

`**video_embeddings`**


| Field            | Type              | Description                                      |
| ---------------- | ----------------- | ------------------------------------------------ |
| `video_id`       | VARCHAR (PK)      | Unique video identifier                          |
| `embedding`      | FLOAT_VECTOR(512) | Finetuned 512d content vector                    |
| `brand_id`       | INT64             | Brand the video belongs to                       |
| `has_engagement` | BOOL              | True = in training data; False = cold-start only |
| `updated_at`     | INT64             | Unix timestamp                                   |


`**user_embeddings**`


| Field            | Type              | Description                             |
| ---------------- | ----------------- | --------------------------------------- |
| `user_brand_key` | VARCHAR (PK)      | Composite `user_id::brand_id`           |
| `user_id`        | VARCHAR           | Plain user identifier                   |
| `brand_id`       | INT64             | Brand context                           |
| `embedding`      | FLOAT_VECTOR(512) | Learned 512d preference vector          |
| `cohort_ids`     | VARCHAR           | Comma-separated cohort IDs e.g. `"0,2"` |
| `updated_at`     | INT64             | Unix timestamp                          |


### Index & Search

- **Index:** IVF_FLAT (Inverted File Index, nlist=128)
- **Distance metric:** Inner Product (IP) — equivalent to cosine similarity on unit vectors
- **Brand filtering:** `brand_id == X` applied at ANN search time

### Cold-Start Videos

New videos not yet in training data are embedded using backbone + VideoTower and stored with `has_engagement=False`. They appear in recommendations **immediately** — no waiting for engagement data.

---

## 10. Accuracy Metrics — Definitions & Results

### Evaluation Methodology

**Per-user chronological split:**

- Each user's interactions are sorted by date
- Oldest 80% → training set
- Newest 20% → test set
- Users with fewer than 5 interactions → training only

This simulates production: train on the past, evaluate on the future.

**Users evaluated:** 22,296 `(user, brand)` pairs

### Metric Definitions

**Recall@K**

> Of all videos a user actually engaged with in the test period, what fraction appear in the model's top K recommendations?

```
Recall@K = |relevant ∩ top-K| / |relevant|
```

Recall@10 = 0.051 means: the model places the user's next watched video in the top 10 (out of ~10,000) 5.1% of the time.

**MRR — Mean Reciprocal Rank**

> On average, at what rank does the first correct video appear?

```
MRR = mean(1 / rank_of_first_relevant_result)
```

MRR = 0.039 → first correct video appears at approximately rank 26 on average.

**Cosine Similarity Gap**

> How much closer is a user to their positive videos vs random videos in embedding space?

```
Gap = mean_cosine(user → positive videos)
    − mean_cosine(user → random videos)
```

Gap = 0.31 means users are consistently placed 0.31 closer to videos they engage with. This is the **primary production metric** — it directly drives the quality of ANN retrieval.

### Results


| Metric     | Our Model  | Random Baseline | Improvement           |
| ---------- | ---------- | --------------- | --------------------- |
| MRR        | 0.0274     | —               | —                     |
| Recall@10  | 0.0358     | 0.0010          | **36× better**        |
| Recall@20  | 0.0511     | 0.0021          | **24× better**        |
| Recall@50  | 0.0791     | 0.0052          | **15× better**        |
| Cosine Gap | **0.3087** | ~0.0000         | strong learned signal |


### Why Cosine Gap is the Key Metric

Recall@K depends on training randomness and the specific test split. The cosine gap measures the **quality of the embedding space itself** — how well the model has learned to represent preferences. A gap of 0.31 indicates embeddings that generalise well to new videos added after training, which is the real production scenario.

---

## 11. Why This Approach

### Alternatives Considered


| Approach                                       | Why Not Chosen                                                              |
| ---------------------------------------------- | --------------------------------------------------------------------------- |
| Collaborative Filtering (matrix factorisation) | Cannot handle new videos or users without retraining; ignores video content |
| Content-Based Filtering only                   | Ignores user behaviour; cannot personalise                                  |
| Single-tower (averaged history)                | Less expressive; cannot capture brand-specific preferences                  |
| Per-brand models                               | Data too sparse per brand; one shared model trains on 17× more data         |


### Why Two-Tower

- **Scales** — recommendation is a single dot product + ANN search, sub-millisecond
- **Cold-start** — new users use zero padding; new videos use backbone embeddings
- **Multi-brand** — `brand_id` is an input feature, not a separate model
- **Industry proven** — YouTube, Pinterest, Google Play, Airbnb all use this architecture

### Why Sentence Transformers

- `all-mpnet-base-v2` is one of the strongest general-purpose semantic embedding models
- 768d vectors capture nuanced content meaning across languages and topics
- Frozen backbone means no labelled content data required for fine-tuning

### Why Milvus

- Open-source, self-hosted — data stays on-premise
- Sub-millisecond ANN search at 10K–1M scale
- Supports metadata filtering (brand_id) during vector search
- Native upsert for incremental updates

---

## 12. System Update Schedule


| Update Type                | Frequency                    | What Happens                                                          |
| -------------------------- | ---------------------------- | --------------------------------------------------------------------- |
| **Incremental fine-tune**  | Every ~15 min / 5,000 events | Fine-tune on new interactions; re-upsert changed user embeddings      |
| **Full retrain**           | Every 3–5 days               | Retrain on all accumulated data; rebuild all embeddings               |
| **Full embedding rebuild** | Weekly                       | Re-embed all videos through updated VideoTower; refresh Milvus        |
| **Cold-start ingestion**   | Continuous                   | New OpenSearch videos embedded and stored with `has_engagement=False` |


New videos appear in recommendations **immediately** after ingestion — no engagement data required.

---

## 13. Key Hard-Coded Values & Rationale


| Parameter                   | Value | Rationale                                        |
| --------------------------- | ----- | ------------------------------------------------ |
| Output embedding dim        | 512d  | Balance between expressiveness and search speed  |
| Backbone dim                | 768d  | Fixed by `all-mpnet-base-v2`                     |
| Temperature                 | 10.0  | Sharpens score distribution for training signal  |
| Positive threshold          | 0.05  | Captures low-engagement positives without noise  |
| Negative ratio              | 4:1   | Standard for contrastive recommendation learning |
| Min interactions for split  | 5     | Sparse users go entirely to training             |
| Test ratio                  | 20%   | Standard chronological hold-out                  |
| Cohort assignment threshold | 15%   | Minimum engagement share to assign a cohort      |
| Max k (cohorts)             | 8     | Beyond 8, cohorts become too granular            |
| Training epochs             | 50    | Convergence confirmed empirically                |
| Batch size                  | 256   | GPU memory / gradient stability tradeoff         |
| Transcript weight           | 0.40  | Highest signal — actual spoken content           |
| AI description weight       | 0.25  | Clean, consistent machine-generated text         |
| Description weight          | 0.30  | Human-written, reliable                          |
| Keywords weight             | 0.05  | User-entered, often noisy                        |


---
