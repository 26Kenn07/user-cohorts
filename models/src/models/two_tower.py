import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

ENGAGEMENT_DIM = 5   # watch_ratio, views (capped), likes, shares, comments


class VideoTower(nn.Module):
    """
    Frozen sentence-transformer backbone + trainable MLP projection.
    The backbone produces rich semantic embeddings — we only finetune the MLP.
    """
    def __init__(self, backbone_dim: int = 384, output_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(backbone_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.mlp(x), dim=-1)


class UserTower(nn.Module):
    """
    YouTube DNN-style user tower.
    Inputs:
      - user_idx:           learnable per-user embedding (0 = unknown/new user)
      - brand_idx:          learnable per-brand embedding
      - engagement_features: raw signals (watch_ratio, views, likes, shares, comments)

    New users get user_idx=0 (padding_idx) — model falls back to brand + engagement signals.
    Returning users get their learned preference embedding.
    """
    def __init__(
        self,
        n_users: int,
        n_brands: int,
        user_embed_dim: int = 64,
        brand_embed_dim: int = 32,
        output_dim: int = 128,
    ):
        super().__init__()
        # padding_idx=0 → new users get zero embedding, not updated during training
        self.user_embed  = nn.Embedding(n_users + 1, user_embed_dim, padding_idx=0)
        self.brand_embed = nn.Embedding(n_brands, brand_embed_dim)
        self.engagement_fc = nn.Sequential(
            nn.Linear(ENGAGEMENT_DIM, 32),
            nn.ReLU(),
        )

        mlp_input_dim = user_embed_dim + brand_embed_dim + 32
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
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
        backbone_dim: int = 384,
        output_dim: int = 128,
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
    Maps string user_ids and brand_ids to integer indices for embedding lookups.
    user_idx=0 is reserved for unknown/new users.
    """
    def __init__(self, df: pd.DataFrame):
        users  = df["user_id"].astype(str).unique().tolist()
        brands = df["brand_id"].astype(str).unique().tolist()

        # 0 = unknown user, known users start at 1
        self.user_to_idx:  dict[str, int] = {u: i + 1 for i, u in enumerate(users)}
        self.brand_to_idx: dict[str, int] = {b: i     for i, b in enumerate(brands)}

        self.n_users  = len(users)
        self.n_brands = len(brands)

    def get_user_idx(self, user_id: str) -> int:
        return self.user_to_idx.get(str(user_id), 0)   # 0 = new/unknown user

    def get_brand_idx(self, brand_id: str) -> int:
        return self.brand_to_idx.get(str(brand_id), 0)


class EngagementDataset(Dataset):
    """
    Each sample: (user_idx, brand_idx, engagement_features, video_embedding, label)

    Positives: rows with score >= positive_threshold
    Negatives: random unwatched videos per user, negative_ratio per positive
    """
    def __init__(
        self,
        engagement_df: pd.DataFrame,
        index_maps: IndexMaps,
        video_embeddings: dict[str, np.ndarray],
        positive_threshold: float = 0.05,
        negative_ratio: int = 4,
    ):
        self.samples: list[tuple] = []
        all_video_ids = list(video_embeddings.keys())

        for user_id, group in engagement_df.groupby("user_id"):
            user_idx  = index_maps.get_user_idx(str(user_id))
            brand_idx = index_maps.get_brand_idx(str(group["brand_id"].iloc[0]))
            watched   = set(group["video_id"].tolist())

            positives = group[group["score"] >= positive_threshold]
            for _, row in positives.iterrows():
                vid_emb = video_embeddings.get(row["video_id"])
                if vid_emb is None:
                    continue
                eng = self._engagement_features(row)
                self.samples.append((user_idx, brand_idx, eng, vid_emb, 1.0))

            # Hard negatives: unwatched videos
            unwatched = [v for v in all_video_ids if v not in watched]
            n_neg = min(len(positives) * negative_ratio, len(unwatched))
            if n_neg == 0:
                continue
            neg_ids = np.random.choice(unwatched, size=n_neg, replace=False)

            # For negatives, use the user's average engagement as context
            avg_eng = self._avg_engagement(group)
            for vid_id in neg_ids:
                vid_emb = video_embeddings.get(vid_id)
                if vid_emb is None:
                    continue
                self.samples.append((user_idx, brand_idx, avg_eng, vid_emb, 0.0))

        n_pos = sum(1 for *_, label in self.samples if label == 1.0)
        n_neg = sum(1 for *_, label in self.samples if label == 0.0)
        logger.info(f"Dataset: {len(self.samples)} samples ({n_pos} pos, {n_neg} neg)")

    @staticmethod
    def _engagement_features(row: pd.Series) -> np.ndarray:
        return np.array([
            float(row["watch_percentage"]) / 100.0,
            min(float(row["views"]), 5.0) / 5.0,   # cap at 5, normalize
            float(row["likes"]),
            float(row["shares"]),
            float(row["comments"]),
        ], dtype=np.float32)

    @staticmethod
    def _avg_engagement(group: pd.DataFrame) -> np.ndarray:
        return np.array([
            float(group["watch_percentage"].mean()) / 100.0,
            min(float(group["views"].mean()), 5.0) / 5.0,
            float(group["likes"].mean()),
            float(group["shares"].mean()),
            float(group["comments"].mean()),
        ], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        user_idx, brand_idx, eng, vid_emb, label = self.samples[idx]
        return (
            torch.tensor(user_idx,  dtype=torch.long),
            torch.tensor(brand_idx, dtype=torch.long),
            torch.tensor(eng,       dtype=torch.float32),
            torch.tensor(vid_emb,   dtype=torch.float32),
            torch.tensor(label,     dtype=torch.float32),
        )


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(
    model: TwoTowerModel,
    dataset: EngagementDataset,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
) -> list[float]:
    device = get_device()
    logger.info(f"Training on {device}")

    model = model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    # Cosine LR decay — helps with convergence on larger datasets
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    epoch_losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for user_idx, brand_idx, eng, vid_emb, labels in loader:
            user_idx  = user_idx.to(device)
            brand_idx = brand_idx.to(device)
            eng       = eng.to(device)
            vid_emb   = vid_emb.to(device)
            labels    = labels.to(device)

            optimizer.zero_grad()
            scores = model(user_idx, brand_idx, eng, vid_emb)
            loss = loss_fn(scores, labels)
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
