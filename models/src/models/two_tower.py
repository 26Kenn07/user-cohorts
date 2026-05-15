import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

ENGAGEMENT_DIM = 6   # watch_ratio, views (capped), likes, shares, comments, link_clicks


class VideoTower(nn.Module):
    """
    Frozen sentence-transformer backbone + trainable MLP projection.
    The backbone produces rich semantic embeddings — we only finetune the MLP.
    """
    def __init__(self, backbone_dim: int = 768, output_dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(backbone_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mlp(x), dim=-1)


class UserTower(nn.Module):
    """
    YouTube DNN-style user tower.
    Inputs:
      - user_idx:            learnable per-user embedding (0 = unknown/new user)
      - brand_idx:           learnable per-brand embedding
      - engagement_features: raw signals (watch_ratio, views, likes, shares, comments)

    New users get user_idx=0 (padding_idx) — model falls back to brand + engagement signals.
    Returning users get their learned preference embedding.
    """
    def __init__(
        self,
        n_users: int,
        n_brands: int,
        user_embed_dim: int = 128,
        brand_embed_dim: int = 64,
        output_dim: int = 512,
    ):
        super().__init__()
        # padding_idx=0 → new users get zero embedding, not updated during training
        self.user_embed    = nn.Embedding(n_users + 1, user_embed_dim, padding_idx=0)
        self.brand_embed   = nn.Embedding(n_brands, brand_embed_dim)
        self.engagement_fc = nn.Sequential(
            nn.Linear(ENGAGEMENT_DIM, 64),
            nn.ReLU(),
        )

        mlp_input_dim = user_embed_dim + brand_embed_dim + 64
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(
        self,
        user_idx: torch.Tensor,
        brand_idx: torch.Tensor,
        engagement: torch.Tensor,
    ) -> torch.Tensor:
        u = self.user_embed(user_idx)
        b = self.brand_embed(brand_idx)
        e = self.engagement_fc(engagement)
        return F.normalize(self.mlp(torch.cat([u, b, e], dim=-1)), dim=-1)


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_brands: int,
        backbone_dim: int = 768,
        output_dim: int = 512,
        temperature: float = 10.0,
    ):
        super().__init__()
        self.user_tower  = UserTower(n_users, n_brands, output_dim=output_dim)
        self.video_tower = VideoTower(backbone_dim, output_dim)
        self.temperature = temperature

    def forward(
        self,
        user_idx: torch.Tensor,
        brand_idx: torch.Tensor,
        engagement: torch.Tensor,
        video_emb: torch.Tensor,
    ) -> torch.Tensor:
        user_out  = self.user_tower(user_idx, brand_idx, engagement)
        video_out = self.video_tower(video_emb)
        return (user_out * video_out).sum(dim=-1) * self.temperature


class IndexMaps:
    """
    Maps (user_id, brand_id) pairs and brand_ids to integer indices.
    A user active on 3 brands gets 3 separate embedding slots.
    user_idx=0 is reserved for unknown/new (user, brand) pairs.
    """
    def __init__(self, df: pd.DataFrame):
        pairs = (
            df[["user_id", "brand_id"]]
            .astype(str)
            .drop_duplicates()
            .reset_index(drop=True)
        )
        brands = df["brand_id"].astype(str).unique().tolist()

        # 0 = unknown pair, known pairs start at 1
        self.pair_to_idx:  dict[tuple[str, str], int] = {
            (row.user_id, row.brand_id): i + 1
            for i, row in enumerate(pairs.itertuples(index=False))
        }
        self.brand_to_idx: dict[str, int] = {b: i for i, b in enumerate(brands)}

        self.n_users  = len(pairs)    # = number of (user, brand) pairs
        self.n_brands = len(brands)

    def get_user_idx(self, user_id: str, brand_id: str) -> int:
        return self.pair_to_idx.get((str(user_id), str(brand_id)), 0)

    def get_brand_idx(self, brand_id: str) -> int:
        return self.brand_to_idx.get(str(brand_id), 0)


class EngagementDataset(Dataset):
    """
    Each sample: (user_idx, brand_idx, engagement_features, pos_vid_emb, neg_vid_embs)

    Structured for InfoNCE loss — each positive is paired with negative_ratio negatives.
    Positives: rows with score >= positive_threshold
    Negatives: hard_negative_ratio semantically similar (backbone cosine) unwatched videos
               + (negative_ratio - hard_negative_ratio) random unwatched videos.
    """
    def __init__(
        self,
        engagement_df: pd.DataFrame,
        index_maps: IndexMaps,
        video_embeddings: dict[str, np.ndarray],
        positive_threshold: float = 0.1,
        negative_ratio: int = 4,
        hard_negative_ratio: int = 2,
    ):
        self.samples: list[tuple] = []
        self.negative_ratio = negative_ratio
        all_video_ids = list(video_embeddings.keys())

        # Precompute normalized video matrix for fast cosine similarity
        vid_matrix = np.stack([video_embeddings[v] for v in all_video_ids]).astype(np.float32)
        norms = np.linalg.norm(vid_matrix, axis=1, keepdims=True)
        vid_matrix_norm = vid_matrix / (norms + 1e-8)
        vid_to_idx = {v: i for i, v in enumerate(all_video_ids)}

        n_hard   = min(hard_negative_ratio, negative_ratio)
        n_random = negative_ratio - n_hard

        for (user_id, brand_id), group in engagement_df.groupby(["user_id", "brand_id"]):
            user_idx  = index_maps.get_user_idx(str(user_id), str(brand_id))
            brand_idx = index_maps.get_brand_idx(str(brand_id))
            watched   = set(group["video_id"].tolist())

            unwatched_mask = np.ones(len(all_video_ids), dtype=bool)
            for v in watched:
                if v in vid_to_idx:
                    unwatched_mask[vid_to_idx[v]] = False
            unwatched_indices = np.where(unwatched_mask)[0]

            if len(unwatched_indices) < negative_ratio:
                continue

            positives = group[group["score"] >= positive_threshold]
            for _, row in positives.iterrows():
                pos_emb = video_embeddings.get(row["video_id"])
                if pos_emb is None:
                    continue

                eng = self._engagement_features(row)

                if n_hard > 0:
                    # Hard negatives: unwatched videos most similar to positive in backbone space
                    pos_norm    = pos_emb / (np.linalg.norm(pos_emb) + 1e-8)
                    sims        = vid_matrix_norm[unwatched_indices] @ pos_norm
                    hard_local  = np.argpartition(sims, -n_hard)[-n_hard:]
                    hard_global = unwatched_indices[hard_local]
                    remaining_mask = np.ones(len(unwatched_indices), dtype=bool)
                    remaining_mask[hard_local] = False
                    remaining_indices = unwatched_indices[remaining_mask]
                else:
                    hard_global       = np.array([], dtype=np.int64)
                    remaining_indices = unwatched_indices

                if len(remaining_indices) < n_random:
                    continue
                random_local  = np.random.choice(len(remaining_indices), size=n_random, replace=False)
                random_global = remaining_indices[random_local]

                neg_idx  = np.concatenate([hard_global, random_global]) if n_hard > 0 else random_global
                neg_embs = vid_matrix[neg_idx]  # already float32

                self.samples.append((user_idx, brand_idx, eng, pos_emb, neg_embs))

        logger.info(
            f"Dataset: {len(self.samples)} samples "
            f"(InfoNCE, {n_hard} hard + {n_random} random negs each)"
        )

    @staticmethod
    def _engagement_features(row: pd.Series) -> np.ndarray:
        return np.array([
            float(row["watch_percentage"]) / 100.0,
            min(float(row["views"]), 5.0) / 5.0,
            float(row["likes"]),
            float(row["shares"]),
            float(row["comments"]),
            float(row.get("link_clicks", 0.0)),
        ], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        user_idx, brand_idx, eng, pos_emb, neg_embs = self.samples[idx]
        return (
            torch.tensor(user_idx,  dtype=torch.long),
            torch.tensor(brand_idx, dtype=torch.long),
            torch.tensor(eng,       dtype=torch.float32),
            torch.tensor(pos_emb,   dtype=torch.float32),
            torch.tensor(neg_embs,  dtype=torch.float32),  # (n_neg, dim)
        )


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def info_nce_loss(
    user_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_embs: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE loss: treats the (user, pos_video) pair as the positive and
    (user, neg_video_i) pairs as negatives.  Cross-entropy over 1+N_neg logits.
    Temperature=0.07 matches SimCLR/MoCo defaults — produces well-separated embeddings.
    """
    # pos_score: (B,1)  neg_scores: (B, n_neg)
    pos_score  = (user_emb * pos_emb).sum(dim=-1, keepdim=True) / temperature
    neg_scores = torch.bmm(neg_embs, user_emb.unsqueeze(-1)).squeeze(-1) / temperature
    logits = torch.cat([pos_score, neg_scores], dim=-1)          # (B, 1+n_neg)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


def train(
    model: TwoTowerModel,
    dataset: EngagementDataset,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    temperature: float = 0.07,
) -> list[float]:
    device = get_device()
    logger.info(f"Training on {device}")

    model = model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Cosine LR decay — helps with convergence on larger datasets
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    epoch_losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for user_idx, brand_idx, eng, pos_emb, neg_embs in loader:
            user_idx  = user_idx.to(device)
            brand_idx = brand_idx.to(device)
            eng       = eng.to(device)
            pos_emb   = pos_emb.to(device)
            neg_embs  = neg_embs.to(device)   # (B, n_neg, dim)

            optimizer.zero_grad()
            user_out = model.user_tower(user_idx, brand_idx, eng)   # (B, dim)
            pos_out  = model.video_tower(pos_emb)                    # (B, dim)

            B, n_neg, in_dim = neg_embs.shape
            neg_out = model.video_tower(neg_embs.view(B * n_neg, in_dim)).view(B, n_neg, -1)

            loss = info_nce_loss(user_out, pos_out, neg_out, temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)
        epoch_losses.append(avg_loss)
        logger.info(f"Epoch {epoch + 1}/{epochs} — loss: {avg_loss:.4f}  lr: {scheduler.get_last_lr()[0]:.2e}")

    return epoch_losses


def get_video_embeddings_finetuned(
    model: TwoTowerModel,
    video_embeddings: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Projects all video backbone embeddings through the finetuned video tower."""
    device = next(model.parameters()).device
    model.eval()
    result = {}
    with torch.no_grad():
        for vid_id, emb in video_embeddings.items():
            t = torch.tensor(emb, dtype=torch.float32).unsqueeze(0).to(device)
            result[vid_id] = model.video_tower(t).squeeze(0).cpu().numpy()
    return result


def get_user_embedding(
    model: TwoTowerModel,
    user_idx: int,
    brand_idx: int,
    engagement: np.ndarray,
) -> np.ndarray:
    """
    Produces a user embedding at inference time.
    Works for both new users (user_idx=0) and returning users.
    """
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        u = torch.tensor([user_idx],  dtype=torch.long).to(device)
        b = torch.tensor([brand_idx], dtype=torch.long).to(device)
        e = torch.tensor(engagement,  dtype=torch.float32).unsqueeze(0).to(device)
        return model.user_tower(u, b, e).squeeze(0).cpu().numpy()
