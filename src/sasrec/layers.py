"""SASRec building blocks (Kang & McAuley, 2018) — pre-norm residual, ReLU FFN."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """Self-attention with learnable WQ, WK, WV (Eq. 2–3)."""

    def __init__(self, embed_dim: int, dropout: float):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        # x: (B, L, D)
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        scale = math.sqrt(self.embed_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # causal: mask future positions
        scores = scores.masked_fill(attn_mask, torch.finfo(scores.dtype).min)
        # padding keys
        scores = scores.masked_fill(key_padding_mask.unsqueeze(1), torch.finfo(scores.dtype).min)

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        return torch.matmul(attn, v)


class PointWiseFFN(nn.Module):
    """Eq. 4: ReLU(S W1 + b1) W2 + b2 applied per position."""

    def __init__(self, embed_dim: int, dropout: float):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, embed_dim)
        self.linear2 = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class SASRecBlock(nn.Module):
    """Pre-norm: x + Dropout(g(LayerNorm(x))) for attention and FFN."""

    def __init__(self, embed_dim: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = ScaledDotProductAttention(embed_dim, dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.ffn = PointWiseFFN(embed_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.ln1(x)
        h = self.attn(h, attn_mask, key_padding_mask)
        x = x + self.dropout(h)

        h = self.ln2(x)
        h = self.ffn(h)
        x = x + self.dropout(h)
        return x
