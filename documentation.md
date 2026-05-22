# User Cohort Generation & Personalised Prompts
### Technical Documentation

**Project:** User Cohorts & Suggested Prompts  
**Team:** Genuin Engineering  
**Date:** May 2026

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Our Approach — The Journey](#2-our-approach--the-journey)
3. [How We Understand Video Content](#3-how-we-understand-video-content)
4. [How We Learn User Preferences](#4-how-we-learn-user-preferences)
5. [Building User Cohorts](#5-building-user-cohorts)
6. [Generating Personalised Prompts](#6-generating-personalised-prompts)
7. [Extension — Personalised Video Feed](#7-extension--personalised-video-feed)
8. [Multi-Brand Support](#8-multi-brand-support)
9. [Accuracy & Validation](#9-accuracy--validation)
10. [System Architecture](#10-system-architecture)
11. [Why This Approach](#11-why-this-approach)
12. [Update Schedule](#12-update-schedule)

---

## 1. Problem Statement

### What We Were Asked to Build
Users on the Genuin platform come from different backgrounds and have different interests. Today, every user sees the same suggested prompts regardless of what they actually engage with. The ask was:

> **Generate meaningful user cohorts — groups of users with similar content interests — and use those cohorts to generate personalised suggested prompts for each group.**

A "suggested prompt" is a short text query (e.g. "celebrity red carpet moments", "best SUV under 30 lakh") that surfaces relevant video content when a user opens the app.

### Why This Matters
- A wellness user and a car enthusiast should not see the same prompts
- Relevant prompts drive more engagement — users find content they actually care about faster
- Cohort-level prompts are scalable — you don't need one model per user, just one per cohort

### The Cold-Start Problem
A brand-new user has no history. We still need to assign them to a cohort quickly and show them relevant prompts from day one.

---

## 2. Our Approach — The Journey

### First Instinct: Just Clustewar Users
The simplest approach is to take user engagement data, compute some representation per user, and run K-Means clustering. We tried this — and hit a problem.

> **Problem:** Clustering raw engagement data or simple user averages produced cohorts that were separated by *which brand the user was on*, not by *what content they liked*. A user on iHeart and a user on a car brand were in different clusters even if they both loved celebrity content.

### The Discovery: Two-Tower Model
While exploring better representations, we came across the **Two-Tower Neural Network** — the same architecture used by YouTube and Pinterest for recommendations. The key insight was:

> If we train a model to predict which videos a user will engage with, the learned user embeddings are a far richer representation of preference than any hand-crafted feature. Clustering *these* learned embeddings produces cohorts that truly reflect content taste.

**But there was still a problem with user-based clustering:** the user tower bakes in brand identity as a learned feature, causing users of the same brand to cluster together regardless of content preference.

### The Final Solution: Cluster Videos, Not Users
We flipped the approach:

1. **Cluster videos** by their content embeddings → get content topic clusters
2. **Assign users to clusters** based on what they watched → users inherit topic cohorts
3. A user who watches both celebrity content and car reviews belongs to **both cohorts**

This produces cohorts that cleanly represent content interests — completely independent of which brand the user is on.

### The Bonus Discovery
Once the Two-Tower model was trained to produce good user and video embeddings for cohort generation, wawe realised the **same embeddings could power a personalised feed** — by ranking videos by their dot-product similarity to a user's embedding vector. This is covered in [Section 7](#7-extension--personalised-video-feed) as an extension that comes for free from the same infrastructure.

---

## 3. How We Understand Video Content

To cluster videos by topic, we first need a numerical representation of what each video is about.

### Text Fields Used
Each video in OpenSearch has four text sources:

| Field | Weight | Why |
|---|---|---|
| `transcript` | **0.40** | Actual spoken content — most specific |
| `description_text` | **0.30** | Human-written, reliable |
| `video_gen_description` | **0.25** | AI-generated, clean and consistent |
| `keywords` | **0.05** | User-entered tags — often noisy |

Keywords were given very low weight after observing that user-entered tags like "us logo", "price subject to change", and "amazon" were polluting cluster definitions.

### Backbone Model: `all-mpnet-base-v2`
We use a **frozen sentence transformer** (110M parameters) to convert each text field into a 768-dimensional semantic vector. "Frozen" means we don't retrain this model — it already understands language well enough.

The final video embedding is a weighted average of the four field embeddings, normalised to unit length.

### Fine-Tuning via VideoTower
The raw 768d backbone embedding is then passed through a small trainable neural network (MLP) that projects it to **512 dimensions**. This projection is trained to align video vectors with user preference vectors — so that a user who loves cars ends up close to car videos in the final space.

```
Video text (transcript + description + AI description + keywords)
        ↓
Sentence Transformer (frozen, 768d)
        ↓
VideoTower MLP (trainable, 768d → 512d)
        ↓
512d video content vector (unit normalised)
```

---

## 4. How We Learn User Preferences

### Engagement Scoring
We combine all engagement signals into a single score per `(user, video)` interaction:

```
score = (watch_percentage/100 × 1.0)
      + (min(views, 3)/1    × 0.5)   ← capped at 3 to reduce loop noise
      + (likes               × 3.0)
      + (shares              × 5.0)   ← strongest signal
      + (comments            × 3.0)
```

Shares are weighted highest — a user sharing a video is the strongest possible signal of interest. Scores are normalised **per brand** (each brand's maximum score becomes 1.0), so a single highly-engaged outlier user on one brand cannot collapse scores for all other brands to near zero.

### The User Tower
The user tower is a neural network that takes three inputs and produces a 512d preference vector:

```
Inputs:
  ┌─ user_idx  → learnable embedding (128d) — "what has this user liked before?"
  ├─ brand_idx → learnable embedding (64d)  — "which brand context is this?"
  └─ engagement features → [watch_ratio, views, likes, shares, comments]
                                ↓ Linear(5→64)

Concatenate all → 256d
        ↓
MLP: 256d → 512d → 256d → 512d
        ↓
L2 Normalise → 512d user preference vector
```

**New users** get `user_idx=0` (zero embedding). The model falls back to brand context + engagement features — enough to serve cohort-level recommendations on day one.

### Training
The model is trained using **InfoNCE loss** (the same contrastive objective used by YouTube, SimCLR, and MoCo). For each positive `(user, video)` pair, 4 random unwatched videos are drawn as negatives. The model must identify the positive from the 5-way choice — this directly forces the embeddings apart rather than just learning a relative threshold.

```
For each positive (user, video) interaction:
  Negatives: 4 random unwatched videos
  Loss: cross-entropy over [pos_score, neg1_score, neg2_score, neg3_score, neg4_score]
  Temperature: 0.07  ← sharp separation, same as SimCLR/MoCo
```

- Positives: interactions with per-brand normalised score ≥ 0.05 (~71,000 training samples)
- Optimiser: AdamW + Cosine LR decay, 50 epochs, gradient clipping

**Why InfoNCE over Binary Cross-Entropy:** BCE only learns that positive scores should be higher than negative scores — it does not control the absolute values. In practice BCE-trained models produce cosine similarities of 0.03–0.09 across all pairs, making ANN search unreliable. InfoNCE explicitly maximises the ratio of positive to negative similarity, producing cosine similarities in the 0.4–0.8 range.

---

## 5. Building User Cohorts

### Step 1 — Find the Right Number of Cohorts
We run K-Means clustering on video embeddings for k=3 to k=8 and measure how well-separated the clusters are using the **silhouette score** (range: -1 to +1, higher = better separation):

```
k=3: 0.2993  ← optimal
k=4: 0.2269
k=5: 0.2455
k=6: 0.2514
k=7: 0.2678
k=8: 0.2744
```

k=3 was automatically selected. The InfoNCE-trained embeddings produce tighter, more separable clusters than the previous model — the optimal k dropped from 5 to 3 and the best silhouette score improved from 0.2470 to 0.2993.

### Step 2 — Cluster 9,629 Videos into 3 Content Topics

K-Means groups videos with similar content embeddings together. The 3 resulting clusters are content-coherent — all celebrity content lands in one cluster, event coverage in another, and so on.

### Step 3 — Assign Users to Cohorts
For each user, we look at what they watched and calculate what fraction of their total engagement falls in each content cluster:

```
Example user engagement breakdown:
  Cohort 1 (Glamorous Events): 68% → assigned ✓  (above 15% threshold)
  Cohort 2 (Celebrity News):   25% → assigned ✓  (above 15% threshold)
  Cohort 0 (Events/Talks):      7% → not assigned (below threshold)

Result: this user belongs to cohorts [1, 2]
```

**16% of users (21,498 of 134,594) belong to multiple cohorts** — reflecting genuinely diverse interests. This is a feature, not a limitation.

### Step 4 — Name Each Cohort with Gemini
Rather than displaying raw keywords, we send 15 representative video descriptions from each cluster to **Gemini 2.5 Flash**, which generates a human-readable label and description. This runs once at training time (3 API calls).

### Final Cohorts

| ID | Label | Videos | Description |
|---|---|---|---|
| 0 | Events, Discussions & Presentations | 1,164 | Diverse content featuring public events, conversations, speeches, and media presentations |
| 1 | Glamorous Event Scenes | 2,543 | Celebrity appearances, fashion, and lively interactions at public events |
| 2 | Celebrity News & Scandals | 5,922 | Public figures' personal lives, relationships, events, and serious legal controversies |

---

## 6. Generating Personalised Prompts

Prompt generation works at three levels of personalisation, all powered by **Gemini 2.5 Flash**:

### Level 1 — User Prompts (cohort-based)
Given a `user_id` and `brand_id`, the system looks up the user's cohort(s) in Milvus and generates prompts matching their established content interests.

```bash
uv run suggest.py user <user_id> --brand-id <brand_id>
```

**Input to Gemini:** cohort label + description + top engagement-weighted keywords  
**Output:** 5 search prompts, 3–8 words each

### Level 2 — Video Prompts (content-based)
Given a `video_id`, the system fetches the video's description and transcript from OpenSearch and generates prompts a viewer of that video would naturally search for.

```bash
uv run suggest.py video <video_id>
```

**Input to Gemini:** video description, AI-generated description, keywords  
**Output:** 5 prompts relevant to that specific video's content

### Level 3 — Combined Prompts (user + video)
The most personalised mode: blends the user's long-term cohort interests with the specific video they are currently watching. Ideal for "what to search next" suggestions shown while a video is playing.

```bash
uv run suggest.py user-video <user_id> --brand-id <brand_id> --video-id <video_id>
```

**Input to Gemini:** user cohort context + video content  
**Output:** 5 prompts relevant to both the user's interests and the current video

### Example Prompts by Cohort

**Cohort 0 — Celebrity News & Life**
- "celebrity red carpet fashion moments"
- "Bravo reality show drama"
- "Hollywood interview behind the scenes"

**Cohort 2 — Car Features & Deals**
- "best SUV features 2025"
- "used car dealership review"
- "car interior review luxury"

**Cohort 4 — Empowerment & Expression**
- "TED talk personal growth"
- "mental health tips daily routine"
- "inspiring movement and wellness"

### Personalisation Logic
```
New user opens app
        ↓
Assigned to cohort(s) based on first few interactions
        ↓
suggest.py user → prompts from their cohort(s)
        ↓
User opens a video
        ↓
suggest.py user-video → prompts blending cohort + video context
        ↓
As engagement history grows → cohort assignment refines
        ↓
Prompts become more precise over time
```

For multi-cohort users, context from all their cohorts is passed to Gemini in a single call — it naturally blends the themes.

---

## 7. Extension — Personalised Video Feed

### What This Means
The same 512d vectors we built for cohort assignment can directly power a **personalised video feed**. Instead of just showing cohort-level prompts, we can rank every video individually for each user.

```
User preference vector (512d)  ·  Video content vector (512d)  =  relevance score
```

The video with the highest dot-product score is the most relevant for that specific user. This is an **Approximate Nearest Neighbour (ANN) search** in Milvus — sub-millisecond even across 10,000 videos.

### How It Extends the Cohort Work
No extra training is needed. The vectors produced for cohort generation are the same vectors used for feed ranking:

```
Cohort use:    user vector  →  assign to nearest content cluster
Feed use:      user vector  →  rank all videos by dot-product similarity
```

### Validation Results

| Metric | Our Model | Random | Improvement |
|---|---|---|---|
| Recall@10 | **10.15%** | 0.10% | **101× better** |
| Recall@20 | **15.86%** | 0.21% | **75× better** |
| Recall@50 | **26.96%** | 0.52% | **52× better** |
| Cosine Gap | **0.58** | ~0.00 | strong signal |

> Recall@10 = 10.15% means: the model surfaces the video a user will actually engage with next inside the top 10 results (out of ~10,000 videos) 10% of the time. Random chance would be 0.1%.

### Inference Options Available Today

```bash
# What videos should I show this user? (personalised feed)
uv run infer.py user-to-video <user_id> --brand-id <brand_id>

# What other videos are like this one? (content similarity)
uv run infer.py video-to-video <video_id>

# Which users would most enjoy this video? (creator tools)
uv run infer.py video-to-user <video_id>

# Which users have similar taste to this user? (cohort exploration)
uv run infer.py user-to-user <user_id> --brand-id <brand_id>

# What should this user search for? (suggested prompts)
uv run suggest.py user <user_id> --brand-id <brand_id>

# What prompts relate to this video?
uv run suggest.py video <video_id>

# Prompts matching both user interest and current video
uv run suggest.py user-video <user_id> --brand-id <brand_id> --video-id <video_id>
```

---

## 8. Multi-Brand Support

### The Problem
The same user can be active on multiple brands (e.g. iHear and a car dealership). Their content preferences on each brand may be completely different and must not be mixed.

### The Solution
Every user is identified by a **composite key**: `user_id::brand_id`

- A user on 3 brands gets 3 independent preference vectors
- Recommendations for brand A only consider that user's brand-A history
- Cohort assignments are also per brand — the same user can be in "Celebrity News" on iHear and "Car Features" on an automotive brand

**Scale:** 130,794 unique users × 17 brands = **134,594 user-brand pairs** tracked independently.

---

## 9. Accuracy & Validation

### How We Tested
We used a **per-user chronological split** — train on the oldest 80% of each user's history, test on the newest 20%. This mirrors real production: the model sees the past and must predict the future.

- **22,296 user-brand pairs evaluated**
- Corpus size at inference: ~10,000 videos

### Metric Definitions

**Recall@K** — Of the videos a user engaged with in the test period, how many appear in the top K results?
```
Recall@10 = 5.1% means the model puts the right video in top 10, 5.1% of the time
```

**MRR (Mean Reciprocal Rank)** — On average, at what position does the first correct video appear?
```
MRR = 0.039 → correct video appears around rank 26 on average
```

**Cosine Similarity Gap** — How much closer is a user to videos they liked vs videos they didn't?
```
Gap = 0.31 → users are measurably closer to content they engage with
           → this directly drives cohort quality and feed ranking
```

### Results Summary

| Metric | Model | Random | Lift |
|---|---|---|---|
| Recall@10 | **10.15%** | 0.10% | **101×** |
| Recall@20 | **15.86%** | 0.21% | **75×** |
| Recall@50 | **26.96%** | 0.52% | **52×** |
| MRR | **0.089** | — | correct video at rank ~11 avg |
| Cosine Gap | **0.58** | ~0.00 | strong |

---

## 10. System Architecture

### Data Flow

```
ClickHouse                    OpenSearch
(user events)                 (video metadata)
     │                              │
     ▼                              ▼
Engagement Scoring          Text Embedding
(watch, like, share,        (transcript + description
 comment → score)            + AI desc + keywords)
     │                              │
     └──────────┬───────────────────┘
                ▼
        Two-Tower Model Training
        (50 epochs, BCELoss,
         AdamW + CosineAnnealingLR)
                │
        ┌───────┴────────┐
        ▼                ▼
   User Vectors      Video Vectors
   (512d per         (512d per
    user×brand)       video)
        │                │
        │         K-Means Clustering
        │         (k=5, silhouette)
        │                │
        │         Video Cohort Map
        │                │
        ▼                ▼
   Cohort Assignment (user → cohorts[])
        │
        ├──► Gemini 2.5 Flash → Cohort Labels
        └──► Claude Sonnet   → Cohort-level Prompts
                │
                ▼
           Milvus Vector DB
      (video_embeddings + user_embeddings)
                │
          ┌─────┴──────────────────────────┐
          ▼                                ▼
   infer.py                          suggest.py
   (ANN search)                      (Gemini 2.5 Flash)
   user→video feed                   user mode   → cohort-based prompts
   video→video similar               video mode  → content-based prompts
   video→user targeting              user+video  → blended prompts
```

### Storage
| Collection | Records | Dimension | Use |
|---|---|---|---|
| `video_embeddings` | 9,629 | 512d | Feed ranking, cohort membership |
| `user_embeddings` | 134,594 | 512d | Feed ranking, cohort assignment |

### Model Checkpoint
`cache/two_tower.pt` — saved after training. Loaded by `suggest.py` and `infer.py` at inference time via Milvus (embeddings are pre-computed and stored; the model itself is not needed at query time).

---

## 11. Why This Approach

### Alternatives We Considered

| Alternative | Problem |
|---|---|
| Simple engagement averaging per user | Brand signal dominated — cohorts split by brand, not interest |
| TF-IDF keyword clustering | Keyword noise ("us logo", generic tags) created meaningless clusters |
| Collaborative filtering | Cannot handle new videos; ignores content meaning |
| Per-brand separate models | Too little data per brand; misses cross-brand user behaviour |
| User embedding clustering | Brand embedding in user tower caused brand-geography clustering |
| BCE loss for two-tower training | Only learns relative ordering — produces cosine sims of 0.03–0.09, making ANN search unreliable |
| Global score normalisation | One outlier user (score 246×) collapses all other users' scores to near zero, starving the training dataset |

### Why Video-Based Cohorts Win
Clustering videos by content (not users) produces topic-pure clusters because:
- Video content doesn't change per brand
- No brand signal in video embeddings
- Users are then assigned to content topics, not brand territories

### Why Two-Tower for Embeddings
- Learns user preferences from actual engagement — far richer than hand-crafted features
- The same embeddings serve both cohort assignment and feed ranking
- Handles new users (zero embedding falls back to brand + engagement features)
- Scales to millions of users and videos without retraining

### Why InfoNCE over BCE
- InfoNCE directly maximises the ratio of positive to negative similarity — embeddings are pulled apart, not just ranked
- Produces cosine similarities in the 0.4–0.8 range (vs 0.03–0.09 with BCE), making ANN search in Milvus meaningful
- Temperature=0.07 provides sharp gradient signal per sample, learning efficiently even from modest datasets

---

## 12. Update Schedule

| Trigger | Action |
|---|---|
| New user interaction | User embedding updated in Milvus via incremental fine-tune |
| Every ~15 min / 5,000 events | Incremental fine-tune on new interactions |
| Every 3–5 days | Full retrain on all accumulated data |
| Weekly | Full video re-embedding through updated VideoTower |
| New video added to OpenSearch | Embedded immediately, stored in Milvus as `has_engagement=False`, appears in feed right away |
| Cohort refresh | Re-run after full retrain — labels regenerated via Gemini |

New videos are available in the feed and in cohort-based prompt generation **immediately** after ingestion, without waiting for any engagement data.

---

*Document generated: May 2026 — Genuin Engineering*
