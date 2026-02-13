"""Embedding modules."""
import torch
import torch.nn.functional as F
from torch import nn


class ConditionalEmbedding(nn.Module):
    """Conditional embedding module for class labels."""

    def __init__(self, num_labels: int, d_model: int, dim: int):
        assert d_model % 2 == 0
        super().__init__()
        self.condEmbedding = nn.Sequential(
            nn.Embedding(
                num_embeddings=num_labels + 1, embedding_dim=d_model, padding_idx=0
            ),
            nn.Linear(d_model, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, labels: torch.Tensor, drop_prob: float = 0.0) -> torch.Tensor:
        if self.training and drop_prob > 0.0:
            mask = (
                torch.rand(labels.size(0), 1, device=labels.device) < drop_prob
            ).squeeze(1)
            # set dropped labels to 0 (null)
            labels = labels.masked_fill(mask, 0)

        emb = self.condEmbedding(labels)
        return emb


class ConditionalDINOEmbedding(nn.Module):
    """Conditional embedding module for dino embeddings."""

    def __init__(self, dim: int, d_model: int = 256, dino_dim: int | None = None):
        super().__init__()

        self.continuous = nn.Sequential(
            nn.Linear(dino_dim, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.null = nn.Parameter(torch.zeros(dino_dim))
        nn.init.normal_(self.null, std=0.02)

    def forward(self, x: torch.Tensor, drop_prob: float = 0.0) -> torch.Tensor:
        """Forward pass with optional embedding dropout."""
        x = F.normalize(x, p=2, dim=-1)
        if self.training and drop_prob > 0.0:
            mask = torch.rand(x.size(0), 1, device=x.device) < drop_prob
            x = torch.where(mask, self.null.expand_as(x), x)
        return self.continuous(x)
