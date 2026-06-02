#!/usr/bin/env python3
"""Pre-compute adapter projections for all items and save to disk.

Run this ONCE after training the adapter. It saves:
    checkpoints/projected_embeddings.pt  — shape (n_items, llm_dim)

All 3 injection modes (B/C/D) then just index into this tensor at eval
time — no adapter forward pass at inference, no LLM loaded here.

Example:
    python scripts/cache_projected_embeddings.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adapter import AdapterConfig, EmbeddingAdapter


def main() -> None:
    ckpt_dir = ROOT / "checkpoints"

    # ── Load item embeddings ──────────────────────────────────────────
    emb_path = ckpt_dir / "item_embeddings.pt"
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing {emb_path}. Run train_sasrec.py first.")
    item_emb = torch.load(emb_path, map_location="cpu", weights_only=True).float()
    n_items, sasrec_dim = item_emb.shape
    print(f"[data] {n_items} items, sasrec_dim={sasrec_dim}")

    # ── Load adapter checkpoint ───────────────────────────────────────
    adapter_path = ckpt_dir / "adapter.pt"
    if not adapter_path.exists():
        raise FileNotFoundError(f"Missing {adapter_path}. Run train_adapter.py first.")
    ckpt = torch.load(adapter_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {})
    config = AdapterConfig(
        sasrec_dim=cfg.get("sasrec_dim", sasrec_dim),
        llm_dim=cfg.get("llm_dim", 2048),
        hidden_dim=cfg.get("hidden_dim", 1024),
    )
    adapter = EmbeddingAdapter(config)
    state = ckpt.get("model_state_dict") or ckpt.get("adapter_state_dict")
    adapter.load_state_dict(state)
    adapter.eval()
    print(f"[adapter] {config.sasrec_dim} → {config.hidden_dim} → {config.llm_dim}")
    print(f"[adapter] trained with: {cfg.get('training_method', 'unknown')}")

    # ── Project all items ─────────────────────────────────────────────
    with torch.no_grad():
        projected = adapter.project(item_emb)  # (n_items, llm_dim)

    out_path = ckpt_dir / "projected_embeddings.pt"
    torch.save(projected, out_path)
    print(f"[done] saved {out_path}  shape={tuple(projected.shape)}")
    print(
        "[info] this cache is used by eval_ranking.py for all injection modes "
        "(B=history, C=candidates, D=both)"
    )


if __name__ == "__main__":
    main()
