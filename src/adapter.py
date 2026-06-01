"""Project SASRec item embeddings into Llama space with reconstruction (Phase 2)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class AdapterConfig:
    sasrec_dim: int = 50
    llm_dim: int = 2048
    hidden_dim: int | None = None  # optional bottleneck in decoder


class EmbeddingAdapter(nn.Module):
    """z = W e + b,  ê = D(z).  Minimize ||e - ê||² to preserve collaborative signal."""

    def __init__(self, config: AdapterConfig | None = None):
        super().__init__()
        config = config or AdapterConfig()
        self.config = config
        d_in = config.sasrec_dim
        d_llm = config.llm_dim

        self.projector = nn.Linear(d_in, d_llm)
        dec_hidden = config.hidden_dim or d_in
        self.decoder = nn.Sequential(
            nn.Linear(d_llm, dec_hidden),
            nn.ReLU(),
            nn.Linear(dec_hidden, d_in),
        )

    def project(self, e: torch.Tensor) -> torch.Tensor:
        """e: (..., sasrec_dim) -> z: (..., llm_dim)"""
        return self.projector(e)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (..., llm_dim) -> ê: (..., sasrec_dim)"""
        return self.decoder(z)

    def forward(self, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.project(e)
        e_hat = self.decode(z)
        return z, e_hat

    def reconstruction_loss(self, e: torch.Tensor) -> torch.Tensor:
        _, e_hat = self.forward(e)
        return nn.functional.mse_loss(e_hat, e)
