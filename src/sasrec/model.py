"""SASRec (Kang & McAuley, 2018): causal self-attention + shared MF prediction."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import SASRecBlock


class SASRec(nn.Module):
    def __init__(
        self,
        n_items: int,
        embed_dim: int = 50,
        max_len: int = 200,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_items = n_items
        self.embed_dim = embed_dim
        self.max_len = max_len

        self.item_emb = nn.Embedding(n_items + 1, embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, embed_dim)
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [SASRecBlock(embed_dim, dropout) for _ in range(n_layers)]
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.padding_idx is not None:
                    with torch.no_grad():
                        m.weight[m.padding_idx].zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1
        )

    def encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """(B, L) item ids (0=pad) -> (B, L, D) hidden states F^(b)."""
        bsz, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, -1)

        x = self.item_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        attn_mask = self._causal_mask(seq_len, input_ids.device)
        pad_mask = input_ids.eq(0)

        for block in self.blocks:
            x = block(x, attn_mask, pad_mask)
        return x

    def score_items(self, hidden: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Dot-product scores r_i = h · M_i^T (Eq. 6). hidden (..., D), item_ids (...)."""
        emb = self.item_emb(item_ids)
        return (hidden * emb).sum(dim=-1)

    def bce_loss(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        neg_ids: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Paper Eq. III-E: BCE on positive + one negative per valid timestep."""
        hidden = self.encode(input_ids)
        pos_scores = self.score_items(hidden, target_ids)
        neg_scores = self.score_items(hidden, neg_ids)

        loss = (
            -F.logsigmoid(pos_scores)
            - F.logsigmoid(-neg_scores)
        )
        valid = valid_mask.float()
        denom = valid.sum().clamp(min=1.0)
        return (loss * valid).sum() / denom

    def forward(
        self,
        input_ids: torch.Tensor,
        target_ids: Optional[torch.Tensor] = None,
        neg_ids: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ):
        if target_ids is not None and neg_ids is not None and valid_mask is not None:
            loss = self.bce_loss(input_ids, target_ids, neg_ids, valid_mask)
            return loss, None

        # inference: scores for all items at last position
        hidden = self.encode(input_ids)
        valid = input_ids.ne(0).long()
        last_pos = valid.sum(dim=1).clamp(min=1) - 1
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        seq_repr = hidden[batch_idx, last_pos, :]

        all_emb = self.item_emb.weight
        logits = torch.matmul(seq_repr, all_emb.transpose(0, 1))
        return logits

    def score_candidates(
        self,
        input_ids: torch.Tensor,
        candidate_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Score candidate items at last timestep. candidate_ids: (B, C)."""
        hidden = self.encode(input_ids)
        valid = input_ids.ne(0).long()
        last_pos = valid.sum(dim=1).clamp(min=1) - 1
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        seq_repr = hidden[batch_idx, last_pos, :].unsqueeze(1)
        cand_emb = self.item_emb(candidate_ids)
        return (seq_repr * cand_emb).sum(dim=-1)

    def item_embedding_matrix(self) -> torch.Tensor:
        return self.item_emb.weight[1 : self.n_items + 1].detach().cpu()
