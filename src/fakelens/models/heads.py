"""Lightweight heads trained on frozen-backbone features. All heads output a single logit (fake = 1).

LinearHead      A1: logistic regression on the L2-normalised final embedding (UnivFD-style).
MultiLayerHead  A2-A4: CLS token from every block -> per-layer LayerNorm -> softmax-weighted sum
                over layers (learnable importance) -> MLP. Inspired by RINE (Koutlis & Papadopoulos,
                ECCV 2024). Forensic (low-level) cues live in intermediate layers, while the final
                CLIP embedding is tuned for semantics.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearHead(nn.Module):
    input_key = "embed"

    def __init__(self, embed_dim: int, **_):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 1)
        self.register_buffer("temperature", torch.ones(()))

    def logits(self, embed: torch.Tensor, cls: torch.Tensor | None = None) -> torch.Tensor:
        return self.fc(F.normalize(embed.float(), dim=-1)).squeeze(-1)

    def forward(self, embed=None, cls=None):
        return self.logits(embed, cls) / self.temperature


class MultiLayerHead(nn.Module):
    input_key = "cls"

    def __init__(self, n_layers: int, dim: int, hidden: int = 512, dropout: float = 0.5,
                 layers: list[int] | None = None, **_):
        super().__init__()
        self.layers = list(layers) if layers is not None else list(range(n_layers))
        L = len(self.layers)
        self.register_buffer("layer_idx", torch.tensor(self.layers, dtype=torch.long))
        self.ln_weight = nn.Parameter(torch.ones(L, dim))
        self.ln_bias = nn.Parameter(torch.zeros(L, dim))
        self.alpha = nn.Parameter(torch.zeros(L))  # softmax -> uniform at init
        if hidden and hidden > 0:
            self.mlp = nn.Sequential(nn.Dropout(dropout * 0.5), nn.Linear(dim, hidden), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(hidden, 1))
        else:
            self.mlp = nn.Linear(dim, 1)
        self.register_buffer("temperature", torch.ones(()))

    def layer_weights(self) -> torch.Tensor:
        return torch.softmax(self.alpha, dim=0)

    def pooled(self, cls: torch.Tensor) -> torch.Tensor:
        x = cls.float().index_select(1, self.layer_idx)            # [B, L, D]
        x = F.layer_norm(x, x.shape[-1:]) * self.ln_weight + self.ln_bias
        return (x * self.layer_weights()[None, :, None]).sum(1)     # [B, D]

    def logits(self, embed=None, cls=None) -> torch.Tensor:
        return self.mlp(self.pooled(cls)).squeeze(-1)

    def forward(self, embed=None, cls=None):
        return self.logits(embed, cls) / self.temperature


def build_head(kind: str, n_layers: int, dim: int, embed_dim: int, **kw) -> nn.Module:
    if kind == "linear":
        return LinearHead(embed_dim)
    if kind == "multilayer":
        return MultiLayerHead(n_layers, dim, **kw)
    raise KeyError(kind)
