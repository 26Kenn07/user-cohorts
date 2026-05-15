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

### First Instinct: Just Cluster Users
The simplest approach is to take user engagement data, compute some representation per user, and run K-Means clustering. We tried this — and hit a problem.

> **Problem:** Clustering raw engagement data or simple user averages produced cohorts that were separated by *which brand the user was on*, not by *what content they liked*. A user on iHear and a user on a car brand were in different clusters even if they both loved celebrity content.

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
Once the Two-Tower model was trained to produce good user and video embeddings for cohort generation, we realised the **same embeddings could power a personalised feed** — by ranking videos by their dot-product similarity to a user's embedding vector. This is covered in [Section 7](#7-extension--personalised-video-feed) as an extension that comes for free from the same infrastructure.

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
      + (min(views, 5)/5    × 0.5)
      + (likes               × 3.0)
      + (shares              × 5.0)   ← strongest signal
      + (comments            × 3.0)
```

Shares are weighted highest — a user sharing a video is the strongest possible signal of interest. Scores are normalised per brand so high-activity brands don't dominate.

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
The model is trained to give **high dot-product scores** to `(user, video)` pairs where the user engaged, and **low scores** to random unwatched videos.

- Positives: interactions with engagement score ≥ 0.05
- Negatives: 4 random unwatched videos per positive
- Loss: Binary Cross-Entropy
- Optimiser: AdamW + Cosine LR decay, 50 epochs

---

## 5. Building User Cohorts

### Step 1 — Find the Right Number of Cohorts
We run K-Means clustering on video embeddings for k=3 to k=8 and measure how well-separated the clusters are using the **silhouette score** (range: -1 to +1, higher = better separation):

```
k=3: 0.2263
k=4: 0.2407
k=5: 0.2470  ← optimal
k=6: 0.1691
k=7: 0.1686
k=8: 0.1570
```

k=5 was automatically selected.

### Step 2 — Cluster 9,629 Videos into 5 Content Topics

K-Means groups videos with similar content embeddings together. The 5 resulting clusters are content-coherent — all car videos land in one cluster, all celebrity interviews in another, and so on.

### Step 3 — Assign Users to Cohorts
For each user, we look at what they watched and calculate what fraction of their total engagement falls in each content cluster:

```
Example user engagement breakdown:
  Cohort 0 (Celebrity):   72% → assigned ✓  (above 15% threshold)
  Cohort 2 (Automotive):  21% → assigned ✓  (above 15% threshold)
  Cohort 4 (Empowerment):  7% → not assigned (below threshold)

Result: this user belongs to cohorts [0, 2]
```

**25% of users (34,244 of 134,594) belong to multiple cohorts** — reflecting genuinely diverse interests. This is a feature, not a limitation.

### Step 4 — Name Each Cohort with Gemini
Rather than displaying raw keywords, we send 15 representative video descriptions from each cluster to **Gemini 2.5 Flash**, which generates a human-readable label and description. This runs once at training time (5 API calls).

### Final Cohorts

| ID | Label | Videos | Users | Description |
|---|---|---|---|---|
| 0 | Celebrity News & Life | 1,416 | ~84,000 | Updates on celebrity events, personal lives, and controversies |
| 1 | Life Experiences & Events | 5,084 | ~92,000 | People share personal experiences, emotions, and styles at diverse events |
| 2 | Car Features & Deals | 1,783 | ~17,000 | Showcasing car models, highlighting features, design, interiors, and promotional offers |
| 3 | Public Figure Conversations | 723 | ~18,000 | Interviews and discussions with public figures about their lives, shows, and relationships |
| 4 | Empowerment & Expression | 623 | ~15,000 | Individuals share insights, express themselves, and inspire personal growth |

---

## 6. Generating Personalised Prompts

### How It Works
Each cohort has a well-defined content identity. We use **Claude (Sonnet)** to generate 5 short, natural-language search prompts per cohort — the kind of query a user in that cohort would naturally type.

**Input to Claude:** top engagement-weighted keywords from videos in that cohort  
**Output:** 5 search prompts, 3–8 words each

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
Served prompts from their cohort(s)
        ↓
As engagement history grows → cohort assignment refines
        ↓
Prompts become more precise over time
```

For multi-cohort users, prompts from all their cohorts are shown (interleaved or rotated).

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
| Recall@10 | 5.1% | 0.10% | **51× better** |
| Recall@20 | 6.1% | 0.21% | **29× better** |
| Recall@50 | 8.5% | 0.52% | **16× better** |
| Cosine Gap | **0.31** | ~0.00 | strong signal |

> Recall@10 = 5.1% means: the model surfaces the video a user will actually engage with next inside the top 10 results (out of ~10,000 videos) 5% of the time. Random chance would be 0.1%.

### Inference Options Available Today

```bash
# What videos should I show this user? (personalised feed)
infer.py user-to-video <user_id> --brand-id <brand_id>

# What other videos are like this one? (content similarity)
infer.py video-to-video <video_id>

# Which users would most enjoy this video? (creator tools)
infer.py video-to-user <video_id>

# Which users have similar taste to this user? (cohort exploration)
infer.py user-to-user <user_id> --brand-id <brand_id>
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
| Recall@10 | 5.1% | 0.10% | **51×** |
| Recall@20 | 6.1% | 0.21% | **29×** |
| Recall@50 | 8.5% | 0.52% | **16×** |
| Cosine Gap | 0.31 | ~0.00 | strong |

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
        └──► Claude Sonnet   → Suggested Prompts
                │
                ▼
           Milvus Vector DB
      (video_embeddings + user_embeddings)
                │
                ▼
     Personalised Feed + Suggested Prompts
```

### Storage
| Collection | Records | Dimension | Use |
|---|---|---|---|
| `video_embeddings` | 9,629 | 512d | Feed ranking, cohort membership |
| `user_embeddings` | 134,594 | 512d | Feed ranking, cohort assignment |

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
