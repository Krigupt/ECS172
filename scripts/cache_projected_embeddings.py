#!/usr/bin/env python3
"""Pre-compute adapter projections for all items and save to disk.

Run this ONCE after training each adapter. Saves:
    checkpoints/projected_embeddings.pt         (reconstruction adapter — default)
    checkpoints/projected_embeddings_ranking.pt (ranking adapter — via --out)

All injection modes (C) then just index into this tensor at eval time —
no adapter forward pass at inference, no LLM loaded here.

Example:
    # After train_adapter.py (reconstruction):
    python scripts/cache_projected_embeddings.py

    # After train_adapter_ranking.py (ranking):
    python scripts/cache_projected_embeddings.py \
        --adapter checkpoints/adapter_ranking.pt \
        --out checkpoints/projected_embeddings_ranking.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adapter import AdapterConfig, EmbeddingAdapter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Adapter checkpoint to project from. Default: checkpoints/adapter.pt",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output path for projected embeddings. Default: checkpoints/projected_embeddings.pt",
    )
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_dir = args.checkpoint_dir

    adapter_path = args.adapter or (ckpt_dir / "adapter.pt")
    out_path = args.out or (ckpt_dir / "projected_embeddings.pt")

    # ── Load item embeddings ──────────────────────────────────────────
    emb_path = ckpt_dir / "item_embeddings.pt"
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing {emb_path}. Run train_sasrec.py first.")
    item_emb = torch.load(emb_path, map_location="cpu", weights_only=True).float()
    n_items, sasrec_dim = item_emb.shape
    print(f"[data] {n_items} items, sasrec_dim={sasrec_dim}")

    # ── Load adapter checkpoint ───────────────────────────────────────
    if not adapter_path.exists():
        raise FileNotFoundError(f"Missing {adapter_path}. Run train_adapter.py or train_adapter_ranking.py first.")
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

    print(f"[adapter] {adapter_path.name}  ({cfg.get('training_method', 'unknown')})")
    print(f"[adapter] {config.sasrec_dim} → {config.hidden_dim} → {config.llm_dim}")
    if cfg.get("llm_model"):
        print(f"[adapter] trained against: {cfg['llm_model']}")
    if cfg.get("best_val_hr1") is not None:
        print(f"[adapter] best_val_HR@1={cfg['best_val_hr1']:.4f}")
    elif cfg.get("best_val_ce") is not None:
        print(f"[adapter] best_val_CE={cfg['best_val_ce']:.4f}")

    # ── Project all items ─────────────────────────────────────────────
    with torch.no_grad():
        projected = adapter.project(item_emb)  # (n_items, llm_dim)

    torch.save(projected, out_path)
    print(f"[done] saved {out_path}  shape={tuple(projected.shape)}")


if __name__ == "__main__":
    main()
