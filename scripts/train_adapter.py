#!/usr/bin/env python3
"""Phase 2: Train adapter (SASRec dim -> Llama dim) with reconstruction loss.

Loads frozen item_embeddings.pt from Phase 1 and learns:
    z = W e + b
    ê = D(z)
    L_rec = ||e - ê||²

Example:
    python scripts/train_adapter.py --smoke
    python scripts/train_adapter.py --epochs 100
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adapter import AdapterConfig, EmbeddingAdapter
from src.data import pick_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--sasrec-dim", type=int, default=50)
    p.add_argument("--llm-dim", type=int, default=2048, help="Llama-3.2-1B hidden size")
    p.add_argument("--val-fraction", type=float, default=0.1, help="Item holdout for val MSE")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def eval_mse(model: EmbeddingAdapter, embeddings: torch.Tensor, batch_size: int = 512) -> float:
    model.eval()
    total = 0.0
    n = 0
    for start in range(0, embeddings.size(0), batch_size):
        batch = embeddings[start : start + batch_size]
        loss = model.reconstruction_loss(batch)
        total += loss.item() * batch.size(0)
        n += batch.size(0)
    return total / max(n, 1)


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 20)
        args.batch_size = min(args.batch_size, 256)

    torch.manual_seed(args.seed)
    device = pick_device()
    ckpt_dir = args.checkpoint_dir

    emb_path = ckpt_dir / "item_embeddings.pt"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"Missing {emb_path}. Run: python scripts/train_sasrec.py"
        )

    item_emb = torch.load(emb_path, map_location="cpu", weights_only=True).float()
    if item_emb.shape[1] != args.sasrec_dim:
        args.sasrec_dim = item_emb.shape[1]
        print(f"[data] using sasrec_dim={args.sasrec_dim} from checkpoint")

    n_items = item_emb.size(0)
    print(f"[device] {device}")
    print(f"[data] {n_items} item embeddings, dim={item_emb.shape[1]}")

    # Item-level train/val split (not user-level)
    rng = random.Random(args.seed)
    indices = list(range(n_items))
    rng.shuffle(indices)
    n_val = max(1, int(n_items * args.val_fraction))
    val_idx = set(indices[:n_val])
    train_idx = [i for i in indices if i not in val_idx]

    train_emb = item_emb[train_idx]
    val_emb = item_emb[list(val_idx)]

    train_loader = DataLoader(
        TensorDataset(train_emb),
        batch_size=args.batch_size,
        shuffle=True,
    )

    config = AdapterConfig(sasrec_dim=args.sasrec_dim, llm_dim=args.llm_dim)
    model = EmbeddingAdapter(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(
        f"[train] {len(train_idx)} train items, {n_val} val items | "
        f"{args.sasrec_dim} -> {args.llm_dim} | epochs={args.epochs}"
    )

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_batches = 0
        t0 = time.time()

        for (batch,) in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = model.reconstruction_loss(batch)
            loss.backward()
            optimizer.step()
            running += loss.item()
            n_batches += 1

        train_mse = running / max(n_batches, 1)
        val_mse = eval_mse(model, val_emb.to(device))
        elapsed = time.time() - t0

        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(
            f"epoch {epoch}/{args.epochs}  "
            f"train_mse={train_mse:.6f}  val_mse={val_mse:.6f}  "
            f"best_val={best_val:.6f}  ({elapsed:.1f}s)"
        )

    model.load_state_dict(best_state)
    out_path = ckpt_dir / "adapter.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "sasrec_dim": args.sasrec_dim,
                "llm_dim": args.llm_dim,
                "best_val_mse": best_val,
            },
        },
        out_path,
    )
    print(f"[done] saved {out_path}  (best val_mse={best_val:.6f})")


if __name__ == "__main__":
    main()
