# Approach Document — Personalised Suggested Prompts & Feed Enhancement

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [System Overview](#3-system-overview)
4. [Data Sources](#4-data-sources)
5. [Video Content Understanding](#5-video-content-understanding)
6. [Engagement Scoring](#6-engagement-scoring)
7. [Two-Tower Neural Network](#7-two-tower-neural-network)
8. [InfoNCE Loss — Why and How](#8-infonce-loss--why-and-how)
9. [Training Pipeline](#9-training-pipeline)
10. [User Cohort Construction](#10-user-cohort-construction)
11. [Cohort Labeling via Gemini](#11-cohort-labeling-via-gemini)
12. [Suggested Prompt Generation](#12-suggested-prompt-generation)
13. [Add-On: Personalised Video Feed](#13-add-on-personalised-video-feed)
14. [Vector Store — Milvus](#14-vector-store--milvus)
15. [Inference Interface](#15-inference-interface)
16. [Evaluation & Metrics](#16-evaluation--metrics)
17. [Update & Retraining Schedule](#17-update--retraining-schedule)
18. [Design Decisions & Alternatives Rejected](#18-design-decisions--alternatives-rejected)

---

## 1. Executive Summary

The system learns what kind of content each user engages with, groups users into cohorts by content topic, and uses a language model (Gemini 2.5 Flash) to generate natural-language search prompts that match each cohort's interests. A user interested in celebrity news sees prompts like *"Bravo reality show drama"*; a user interested in automotive content sees *"best SUV features 2025"*.

As a direct by-product of the same infrastructure, the system also enables a **personalised video feed** ranking every video in the catalogue by how well it matches an individual user's preferences. This feed enhancement is an add-on that comes for free from the same embeddings.

**Primary deliverable:** Cohort-based personalised suggested prompts  
**Secondary deliverable:** Individual-level personalised video feed

---

## 2. Problem Statement

### 2.1 The Core Ask

> Generate meaningful user cohorts of users with similar content interests and use those cohorts to generate personalised suggested prompts for each group.

A *suggested prompt* is a short natural-language query (3–8 words) shown to a user when they open the Octo. It surfaces relevant video content and drives engagement. Currently all users see prompts according to the video.

### 2.2 Why This Matters

A wellness user and a car enthusiast should not see the same prompts. Relevant prompts:

- Help users find content they care about faster
- Increase search engagement and session depth
- Are scalable — one set of prompts per cohort, not per user

### 2.3 The Cold-Start Problem

A brand-new user has no history. They still need to be assigned to a cohort quickly and see relevant prompts from their very first session. The architecture handles this explicitly — new users fall back to brand context and immediate engagement signals.

### 2.4 The Multi-Brand Constraint

The same user can be active on multiple brands (e.g. an entertainment platform and a car dealership). Their content preferences on each brand are completely independent and must not be mixed. The system tracks every user as a composite `user_id::brand_id` key.

---

## 3. System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION                              │
│  ClickHouse (user events)      CSV / OpenSearch (video metadata)    │
└──────────────────┬──────────────────────────┬───────────────────────┘
                   │                          │
                   ▼                          ▼
          Engagement Scoring          Video Text Embedding
          (per-brand normalised)      (sentence-transformer 768d)
                   │                          │
                   └──────────┬───────────────┘
                              ▼
                   Two-Tower Model Training
                   (InfoNCE loss, 50 epochs)
                              │
                   ┌──────────┴──────────┐
                   ▼                     ▼
             User Vectors          Video Vectors
             (512d per             (512d per
              user×brand)           video)
                   │                     │
                   │           K-Means Clustering
                   │           (k selected via silhouette)
                   │                     │
                   │            Video Cohort Map
                   │                     │
                   ▼                     ▼
            Cohort Assignment     Cohort Profiles
            (user → cohorts[])    (keywords + labels)
                   │                     │
                   │             Gemini 2.5 Flash
                   │             (cohort labeling)
                   │                     │
                   ▼                     ▼
              Milvus Vector DB   ◄────────────────────
         (video_embeddings +              │
          user_embeddings)                │
                   │                      │
          ┌────────┴──────────────────────┤
          ▼                               ▼
    infer.py                        suggest.py
    Personalised Feed               Suggested Prompts
    (ANN search, add-on)            ← PRIMARY DELIVERABLE
```

---

## 4. Data Sources

### 4.1 ClickHouse — User Events

**Table:** `genuin_events_logs_001`  
**Scale:** ~28 million rows across 17 brand IDs

Each row represents one user interaction with one video on one day. The schema aggregates multiple events into per-`(user_id, video_id, report_date)` counts:


| Column             | Type    | Description                                                          |
| ------------------ | ------- | -------------------------------------------------------------------- |
| `user_id`          | VARCHAR | Platform user identifier                                             |
| `identity_type`    | VARCHAR | `context_ad_id` (device-level global) or `identity_id` (brand-local) |
| `video_id`         | VARCHAR | UUID of the video                                                    |
| `brand_id`         | INT     | Brand the user was on                                                |
| `report_date`      | DATE    | Aggregation date                                                     |
| `views`            | INT     | Number of times the video was loaded                                 |
| `watch_percentage` | FLOAT   | Average % of video duration watched                                  |
| `likes`            | INT     | Number of like events                                                |
| `shares`           | INT     | Number of share events                                               |
| `comments`         | INT     | Number of comment events                                             |


**Event types captured:** `video_impression`, `video_watched`, `video_sparked` (like), `video_shared`, `commented_on_video`

### 4.2 OpenSearch / CSV — Video Metadata

**Index:** `genuin_loop_video_index`  
**Scale:** 9,629 videos with sufficient metadata for embedding


| Field                         | Description                                  |
| ----------------------------- | -------------------------------------------- |
| `transcript`                  | Spoken content from video (most informative) |
| `description_text`            | Human-written description                    |
| `video_gen_description`       | AI-generated content summary                 |
| `keywords.explicit[].keyword` | User-entered tags (noisy)                    |


Videos without any text content are embedded as zero vectors and flagged as low-confidence. In production, video metadata is fetched via `mget` for only the video IDs present in event data.

---

## 5. Video Content Understanding

### 5.1 Backbone: `all-mpnet-base-v2`

We use a **frozen sentence transformer** (110M parameters, 768-dimensional output) to convert each text field into a semantic vector. This model is not retrained — it already encodes general language semantics.

**Why frozen:** Retraining the backbone would require far more data and compute than available, and would risk degrading its general language understanding. The trainable MLP on top is sufficient to adapt its representations to our domain.

### 5.2 Weighted Field Combination

Each video's four text fields are embedded separately, then combined as a weighted average:


| Field                   | Weight   | Rationale                                          |
| ----------------------- | -------- | -------------------------------------------------- |
| `transcript`            | **0.40** | Actual spoken content — most specific and reliable |
| `description_text`      | **0.30** | Human-written, curated signal                      |
| `video_gen_description` | **0.25** | AI-generated, consistent quality                   |
| `keywords`              | **0.05** | User-entered — noisy, error-prone                  |


Keywords were downweighted significantly after observing that user-entered tags like *"us logo"*, *"price subject to change"*, and *"amazon"* were polluting cluster definitions. Each keyword is embedded individually and averaged before combining.

**Combination formula:**

```
weighted_sum = Σ (field_embedding[f] × weight[f])   for non-empty fields
final_weight = Σ weight[f]                           for non-empty fields
video_emb    = L2_normalise(weighted_sum / final_weight)
```

Fields missing from a video are skipped and their weight is redistributed proportionally. The result is a **768-dimensional unit-normalised vector** per video.

### 5.3 VideoTower MLP Projection

The 768d backbone embedding is passed through a trainable MLP that projects it to **512 dimensions**. This projection is learned during Two-Tower training to align video content vectors with user preference vectors in a shared space.

```
VideoTower architecture:
  Linear(768 → 512)
  LayerNorm(512)
  ReLU
  Dropout(0.2)
  Linear(512 → 512)
  L2 Normalise
```

The final output is a **512-dimensional unit-normalised vector** representing "what this video is about, in user-preference space."

---

## 6. Engagement Scoring

### 6.1 Score Formula

Each `(user_id, video_id)` interaction is reduced to a single scalar score:

```
raw_score = (watch_percentage / 100) × 1.0
          + clip(views, max=3)        × 0.5
          + likes                     × 3.0
          + shares                    × 5.0
          + comments                  × 3.0
```

**Signal weights rationale:**

- `watch_percentage`: baseline engagement — did they actually watch?
- `views` capped at 3: re-watches show interest, but looping inflates counts
- `likes` (3×): explicit positive signal
- `shares` (5×): strongest possible signal — the user actively promoted this content
- `comments` (3×): high-intent engagement

### 6.2 Per-Brand Normalisation

Scores are normalised **per brand** (not globally):

```python
df["score"] = df.groupby("brand_id")["score"].transform(
    lambda g: g / g.max() if g.max() > 0 else g
)
```

**Why per-brand:** A single outlier user with extremely high engagement (observed raw score: 246×) would collapse every other user's score to near zero under global normalisation, reducing the training dataset from ~71,000 samples to just 81. Per-brand normalisation ensures that within each brand, the most engaged user gets score=1.0 and all others are scaled relative to that brand's activity level.

---

## 7. Two-Tower Neural Network

The Two-Tower (Dual Encoder) architecture is the same family used by YouTube DNN and Pinterest recommendations. Two separate neural networks encode users and videos into a shared 512-dimensional embedding space. A user and a video are considered a good match if their vectors have high dot-product similarity.

### 7.1 VideoTower

Described in Section 5.3. Takes a 768d backbone embedding, outputs a 512d unit vector.

### 7.2 UserTower

The user tower takes three inputs and produces a 512d preference vector:

```
Inputs:
  user_idx    → Embedding(n_users + 1, 128d)   padding_idx=0
  brand_idx   → Embedding(n_brands, 64d)
  engagement  → [watch_ratio, views_capped, likes, shares, comments]  (5 floats)
                  ↓
              Linear(5 → 64) + ReLU

Concatenate: [128d + 64d + 64d] = 256d
      ↓
MLP:
  Linear(256 → 512)
  LayerNorm(512)
  ReLU
  Dropout(0.2)
  Linear(512 → 256)
  ReLU
  Linear(256 → 512)
  L2 Normalise
```

**User identity encoding:**

- `user_idx=0` is reserved as a padding index (zero vector) for new/unknown users
- Each `(user_id, brand_id)` pair gets a unique embedding slot — a user active on 3 brands has 3 independent preference embeddings
- This is tracked via `IndexMaps`, which is built from training data only
- Test users not seen during training automatically get `user_idx=0` (cold-start)

**Cold-start behaviour:** New users get the zero user embedding. The model falls back to `brand_idx` + `engagement` features — enough to serve cohort-level recommendations immediately.

### 7.3 Similarity Score

At inference time, relevance between user `u` and video `v` is:

```
score(u, v) = dot(user_tower(u), video_tower(v)) × temperature
```

Temperature (default=10.0) sharpens the score distribution. Both towers output L2-normalised vectors, so the dot product equals cosine similarity (range [-1, 1]).

---

## 8. InfoNCE Loss Why and How

### 8.1 Why Not Binary Cross-Entropy

The previous version used BCE loss, which trains the model to output high scores for positive `(user, video)` pairs and low scores for negatives. BCE only constrains the **relative ordering** — it does not control where in the [-1, 1] range the scores land.

In practice, BCE-trained models converge to cosine similarities in the range **0.03–0.09** across all pairs. At this scale, positive and negative pairs are nearly indistinguishable, making ANN search in a vector database unreliable.

### 8.2 InfoNCE (Noise-Contrastive Estimation)

InfoNCE reformulates the training objective as a classification problem: given one positive video and N negative videos, the model must identify which one is the positive. This **explicitly maximises the ratio** of positive to negative similarity, directly producing well-separated embeddings.

```
For each positive (user, video_pos) pair with N=4 negatives:

  pos_score  = dot(user_emb, pos_emb)  / τ
  neg_score_i = dot(user_emb, neg_emb_i) / τ   for i in 1..4

  logits = [pos_score, neg_score_1, neg_score_2, neg_score_3, neg_score_4]
  loss   = CrossEntropy(logits, label=0)   ← label 0 = first position = positive
```

**Temperature τ = 0.07** (same as SimCLR, MoCo). Lower temperature sharpens the distribution — the model is penalised heavily if it scores a negative even close to the positive.

### 8.3 Dataset Construction for InfoNCE

Each training sample is structured as:

```
(user_idx, brand_idx, engagement_features, pos_video_emb, neg_video_embs[4])
```

- **Positives:** interactions with per-brand normalised score ≥ 0.05
- **Negatives:** 4 random videos the user has never watched
- **Dataset size:** ~71,000 samples across 25,653 user-brand pairs
- **Batch collation:** negatives are stacked as a `(B, 4, 768)` tensor; reshaped to `(B×4, 768)` before VideoTower, then back to `(B, 4, 512)` for the loss

**Result:** Cosine similarities shift from 0.03–0.09 (BCE) to **0.4–0.8** (InfoNCE), making vector search meaningful.

---

## 9. Training Pipeline

### 9.1 Configuration


| Hyperparameter          | Value                       |
| ----------------------- | --------------------------- |
| Backbone dim            | 768                         |
| Output dim              | 512                         |
| User embed dim          | 128                         |
| Brand embed dim         | 64                          |
| Engagement FC           | Linear(5→64) + ReLU         |
| Temperature (InfoNCE)   | 0.07                        |
| Temperature (inference) | 10.0                        |
| Negative ratio          | 4                           |
| Positive threshold      | 0.05 (per-brand normalised) |
| Epochs                  | 50                          |
| Batch size              | 256                         |
| Learning rate           | 1e-3                        |
| Weight decay            | 1e-5                        |
| LR schedule             | CosineAnnealingLR           |
| Gradient clip           | max_norm=1.0                |
| Optimiser               | AdamW                       |


### 9.2 Training Loop (per batch)

```python
user_emb = user_tower(user_idx, brand_idx, engagement)    # (B, 512)
pos_emb  = video_tower(pos_video_emb)                      # (B, 512)
neg_emb  = video_tower(neg_videos.view(B*4, 768))
           .view(B, 4, 512)                                # (B, 4, 512)

loss = info_nce_loss(user_emb, pos_emb, neg_emb, τ=0.07)
loss.backward()
clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

### 9.3 Chronological Train/Test Split

To simulate production (model sees the past, predicts the future):

- Each user's interactions are sorted by `report_date`
- Oldest 80% → training set
- Newest 20% → test set
- Users with fewer than 5 interactions go entirely to training (no test split)

**Scale:** 134,594 user-brand pairs, 22,296 evaluated in test set.

### 9.4 Device Selection

```python
if torch.backends.mps.is_available():  return "mps"   # Apple Silicon
if torch.cuda.is_available():          return "cuda"   # NVIDIA GPU
return "cpu"
```

---

## 10. User Cohort Construction

### 10.1 Why Cluster Videos, Not Users

An early attempt clustered user embeddings directly. The result was cohorts split by **which brand the user was on**, not by content interest — because the UserTower bakes in `brand_idx` as a learned feature.

The solution: **cluster videos by content, then assign users based on what they watched**. Video embeddings carry no brand signal — a car review video looks the same regardless of which brand's feed it appeared on.

### 10.2 Step 1 — K-Means on Video Embeddings

Finetuned 512d video vectors (from VideoTower) are clustered using K-Means. The optimal number of clusters k is selected automatically using the **silhouette score**:

```
silhouette(k) = mean over all samples of:
    (b - a) / max(a, b)

where:
    a = mean intra-cluster distance for this sample
    b = mean distance to nearest other cluster
```

Range: [-1, 1]. Higher = better separation. Search range: k=5 to k=8.

**Results from current model:**

```
k=3: 0.2993  ← selected
k=4: 0.2269
k=5: 0.2455
k=6: 0.2514
k=7: 0.2678
k=8: 0.2744
```

k=3 was selected. The InfoNCE-trained embeddings produce tighter clusters (silhouette 0.2993) than the previous BCE model (best was 0.2470 at k=5).

### 10.3 Step 2 — Multi-Cohort User Assignment

For each `(user_id, brand_id)` pair, the system calculates what fraction of their total engagement falls in each video cohort:

```python
for each video the user watched:
    cohort_id = video_cohort_map[video_id]
    cohort_scores[cohort_id] += engagement_score

for each cohort_id:
    share = cohort_scores[cohort_id] / total_score
    if share >= 0.15:
        assign user to this cohort
```

If no cohort clears the 15% threshold (e.g. perfectly uniform engagement), the user is assigned to whichever cohort has the highest score.

**Result:** 134,594 users assigned; **21,498 (16%) belong to multiple cohorts** — users with genuinely diverse interests across content categories. This is a feature, not a limitation.

### 10.4 Cohort Profiles

For each cohort, a profile is built from top engagement-weighted keywords across all videos in the cluster:

```python
for each interaction in engagement_df where video is in cohort:
    for each keyword of that video:
        keyword_score[keyword] += interaction_score

top_keywords = sorted by keyword_score, take top 15
```

---

## 11. Cohort Labeling via Gemini

Raw keywords are noisy. To produce human-readable cohort labels, we send 15 representative video descriptions from each cluster to **Gemini 2.5 Flash** and ask for a label and one-sentence description.

**Prompt structure:**

```
You are categorizing content clusters for a short video platform.

Here are [N] video descriptions from one content cluster:
[1] <video_gen_description or description_text or transcript, max 300 chars>
...

Provide:
1. A short label (2-4 words) capturing the theme
2. A one-sentence description (max 12 words)

Respond in exactly this format:
Label: <label>
Description: <description>
```

This runs once at training time — 3 API calls (one per cohort). Labels are cached and stored in `cohort_profiles.pkl`.

**Current cohorts (May 2026):**


| ID  | Label                               | Videos | Description                                                              |
| --- | ----------------------------------- | ------ | ------------------------------------------------------------------------ |
| 0   | Events, Discussions & Presentations | 1,164  | Diverse public events, conversations, speeches, and media presentations  |
| 1   | Glamorous Event Scenes              | 2,543  | Celebrity appearances, fashion, and lively interactions at public events |
| 2   | Celebrity News & Scandals           | 5,922  | Public figures' personal lives, relationships, events, and controversies |


---

## 12. Suggested Prompt Generation

This is the **primary deliverable**. The system generates 5 natural-language search prompts per request using Gemini 2.5 Flash. Three modes of increasing personalisation:

### 12.1 Mode 1 — User Mode (Cohort-Based)

Looks up the user's cohort(s) from Milvus and generates prompts matching their established content interests.

**Data flow:**

```
user_id + brand_id
    ↓
Milvus query → cohort_ids (e.g. "0,2")
    ↓
Load cohort_profiles.pkl → labels + keywords for each cohort
    ↓
Build context string:
  "A user on a short video platform has these content interests:
   • Events, Discussions & Presentations: [description]
     Top topics: ted talk, bollywood, indian express, ...
   • Celebrity News & Scandals: [description]
     Top topics: us weekly, red carpet, interview, ..."
    ↓
Gemini 2.5 Flash → 5 prompts
```

**CLI:**

```bash
uv run suggest.py user <user_id> --brand-id <brand_id>
```

### 12.2 Mode 2 — Video Mode (Content-Based)

Generates prompts based on a specific video's content. Used for "related searches" shown alongside a video.

**Data flow:**

```
video_id
    ↓
Load cache/videos.pkl → video metadata (description, transcript, keywords)
    ↓
Build context string:
  "A user is watching a video about:
   [video_gen_description or description_text, max 400 chars]
   Keywords: [up to 15 keywords]"
    ↓
Gemini 2.5 Flash → 5 prompts
```

**CLI:**

```bash
uv run suggest.py video <video_id>
```

### 12.3 Mode 3 — User+Video Mode (Blended)

The most personalised mode. Blends the user's long-term cohort interests with the specific video they are currently watching. Ideal for "what to search next" suggestions displayed while a video plays.

**Data flow:**

```
user_id + brand_id + video_id
    ↓
[Both user cohort context AND video content context]
    ↓
Build combined context:
  "User interests: [cohort labels + keywords]
   Currently watching: [video description]
   Generate prompts relevant to BOTH interests AND this video."
    ↓
Gemini 2.5 Flash → 5 prompts (blended)
```

**CLI:**

```bash
uv run suggest.py user-video <user_id> --brand-id <brand_id> --video-id <video_id>
```

### 12.4 Gemini Prompt Template

All three modes share the same generation wrapper:

```
{context}

Generate exactly 5 short, natural search prompts (3–8 words each)
that this user would type into a short video app to find content they'd enjoy.

Rules:
- Be specific to the topics described above
- Write as a user would naturally search (lowercase, conversational)
- Return only the prompts, one per line, no numbering or extra text
```

### 12.5 Personalisation Over Time

```
Day 1 (new user):
    user_idx = 0 (cold-start)
    model falls back to brand + watch signals
    assigned to cohort on first few interactions
    → user mode prompts from initial cohort

Day 7 (returning user):
    user_idx = learned embedding
    cohort assignment stabilises
    → user mode prompts reflect established taste

While watching video X:
    user-video mode blends cohort + video context
    → prompts reflect both interests and current session
```

---

## 13. Add-On: Personalised Video Feed

The same 512d vectors produced for cohort generation can directly rank videos for individual users — no additional training required.

### 13.1 How It Works

```
relevance(user u, video v) = dot(user_emb(u), video_emb(v))
                           = cosine_similarity(u, v)   [both L2-normalised]
```

The video with the highest dot product is the most relevant for that specific user. This is an **Approximate Nearest Neighbour (ANN) search** in Milvus — sub-millisecond even across 10,000 videos.

### 13.2 Four Query Modes (`infer.py`)


| Mode             | Source → Target                    | Use Case                                          |
| ---------------- | ---------------------------------- | ------------------------------------------------- |
| `user-to-video`  | user embedding → video collection  | Personalised feed for a user                      |
| `video-to-video` | video embedding → video collection | "More like this"                                  |
| `video-to-user`  | video embedding → user collection  | Which users should see this video (creator tools) |
| `user-to-user`   | user embedding → user collection   | Users with similar taste                          |


### 13.3 Cold-Start for New Videos

New videos added to OpenSearch are embedded immediately via the backbone + VideoTower and stored in Milvus with `has_engagement=False`. They appear in recommendations right away — no engagement data required.

---

## 14. Vector Store — Milvus

### 14.1 Collections

`**video_embeddings`**


| Field            | Type              | Notes                       |
| ---------------- | ----------------- | --------------------------- |
| `video_id`       | VARCHAR(256)      | Primary key                 |
| `embedding`      | FLOAT_VECTOR(512) | Finetuned VideoTower output |
| `brand_id`       | INT64             | For brand-scoped filtering  |
| `has_engagement` | BOOL              | False = cold-start video    |
| `updated_at`     | INT64             | Unix timestamp              |


`**user_embeddings**`


| Field            | Type              | Notes                                    |
| ---------------- | ----------------- | ---------------------------------------- |
| `user_brand_key` | VARCHAR(512)      | Primary key, format: `user_id::brand_id` |
| `user_id`        | VARCHAR(256)      |                                          |
| `brand_id`       | INT64             |                                          |
| `embedding`      | FLOAT_VECTOR(512) | UserTower output                         |
| `cohort_ids`     | VARCHAR(64)       | Comma-separated, e.g. `"0,2"`            |
| `updated_at`     | INT64             | Unix timestamp                           |


### 14.2 Index Configuration

Both collections use **IVF_FLAT** (Inverted File with Flat quantisation):

```
index_type:  IVF_FLAT
metric_type: IP        ← Inner Product (= cosine similarity for unit vectors)
nlist:       128       ← number of Voronoi cells
nprobe:      128       ← cells scanned at query time (= exact search)
```

`nprobe=128` with `nlist=128` means all cells are scanned — effectively exact search. This is appropriate at our scale (~10K videos, ~135K users). At 1M+ videos, `nprobe` should be tuned down (e.g. 16–32) to trade a small recall hit for speed.

### 14.3 Scale


| Collection         | Records | Dimension    | Memory (approx.) |
| ------------------ | ------- | ------------ | ---------------- |
| `video_embeddings` | 9,629   | 512d float32 | ~20 MB           |
| `user_embeddings`  | 134,594 | 512d float32 | ~275 MB          |


---

## 15. Inference Interface

### 15.1 suggest.py — Prompt Generation

```
uv run suggest.py user        <user_id>  --brand-id <id>
uv run suggest.py video       <video_id>
uv run suggest.py user-video  <user_id>  --brand-id <id>  --video-id <id>
```

**Dependencies at runtime:**

- Milvus (for user/video lookup)
- `cache/videos.pkl` (for video metadata, local)
- `cache/cohort_profiles.pkl` (for cohort labels/keywords, local)
- `GEMINI_API_KEY` environment variable

### 15.2 infer.py — Feed & Similarity Search

```
uv run infer.py user-to-video  <user_id>   --brand-id <id>  [--top-k 20]
uv run infer.py video-to-video <video_id>                   [--top-k 20]
uv run infer.py video-to-user  <video_id>                   [--top-k 20]
uv run infer.py user-to-user   <user_id>   --brand-id <id>  [--top-k 20]
```

---

## 16. Evaluation & Metrics

### 16.1 Evaluation Protocol

**Per-user chronological split:**

- Train on oldest 80% of each user's history
- Test on newest 20%
- Model sees the past, must predict the future

**Scale:** 22,296 user-brand pairs evaluated, corpus of ~9,629 videos.

### 16.2 Metric Definitions

**Recall@K:** Of the videos a user engaged with in the test period, what fraction appear in the top K model results?

```
Recall@10 = 10.15% means: given ~10,000 videos,
the model puts the next video a user will engage with
in the top 10 results, 10.15% of the time.
Random chance: 0.10%.
```

**MRR (Mean Reciprocal Rank):** Average of 1/rank for the first correct result.

```
MRR = 0.089 → correct video appears around rank 11 on average
```

**Cosine Similarity Gap:** Mean difference between cosine similarity to positive videos vs random negatives.

```
Gap = 0.58 → users are measurably and strongly closer
             to content they engage with
```

### 16.3 Results


| Metric     | InfoNCE Model | BCE (prev) | Random | Lift vs Random |
| ---------- | ------------- | ---------- | ------ | -------------- |
| Recall@10  | **10.15%**    | 5.10%      | 0.10%  | **101×**       |
| Recall@20  | **15.86%**    | 6.10%      | 0.21%  | **75×**        |
| Recall@50  | **26.96%**    | 8.50%      | 0.52%  | **52×**        |
| MRR        | **0.089**     | 0.039      | —      | —              |
| Cosine Gap | **0.58**      | 0.31       | ~0.00  | strong         |


The switch from BCE to InfoNCE loss roughly **doubled every metric**.

---

## 17. Update & Retraining Schedule


| Trigger                       | Action                                                                       | Latency    |
| ----------------------------- | ---------------------------------------------------------------------------- | ---------- |
| New video added to OpenSearch | Embed via backbone + VideoTower → upsert to Milvus (`has_engagement=False`)  | < 1 second |
| New user interaction          | Update user embedding via `get_user_embedding()` → upsert to Milvus          | < 100 ms   |
| Every ~15 min / 5,000 events  | Incremental fine-tune on new interactions                                    | ~2 min     |
| Every 3–5 days                | Full retrain on all accumulated data                                         | ~30 min    |
| After full retrain            | Re-run cohort clustering → regenerate labels via Gemini → regenerate prompts | ~5 min     |
| Cohort labels stale           | Re-run `generate_all_labels()` — 3 Gemini API calls                          | < 1 min    |


New videos appear in the feed and in prompt generation **immediately** after ingestion, without waiting for engagement data.

---