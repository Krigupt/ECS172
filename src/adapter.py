"""Project SASRec item embeddings into Llama space (Phase 2).

Redesigned: the projector is trained via teacher-forced cross-entropy through
a frozen LLaMA, *not* via MSE reconstruction.  The decoder is removed — LLaMA
itself acts as the decoder during adapter training.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class AdapterConfig:
    sasrec_dim: int = 50
    llm_dim: int = 2048
    hidden_dim: int = 1024  # intermediate MLP width in projector


class EmbeddingAdapter(nn.Module):
    """z = Projector(e).  Trained so frozen LLaMA can decode movie title from z.

    Architecture (per plan):
        Linear(sasrec_dim → hidden_dim) → SiLU → LayerNorm → Linear(hidden_dim → llm_dim)
    """

    def __init__(self, config: AdapterConfig | None = None):
        super().__init__()
        config = config or AdapterConfig()
        self.config = config
        d_in = config.sasrec_dim
        d_hidden = config.hidden_dim
        d_llm = config.llm_dim

        self.projector = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.SiLU(),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_llm),
        )

    def project(self, e: torch.Tensor) -> torch.Tensor:
        """e: (..., sasrec_dim) -> z: (..., llm_dim)"""
        return self.projector(e)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """Same as project(); kept for nn.Module compatibility."""
        return self.project(e)

    def reconstruction_loss(self, e: torch.Tensor) -> torch.Tensor:
        """Stub — returns 0.  Kept so inject_llm.forward_batch doesn't break
        when rec_lambda > 0 (the regularization term is simply zeroed out)."""
        return torch.tensor(0.0, device=e.device, requires_grad=True)
