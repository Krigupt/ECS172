#!/usr/bin/env python3
"""Phase 3: Fine-tune adapter with Llama frozen (embedding injection + ranking loss).

Example:
    python scripts/finetune_ranking.py --smoke
    python scripts/finetune_ranking.py --epochs 3 --max-train 2000
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import IdMaps, load_interactions, pick_device
from src.inject_llm import InjectedLlamaRanker
from src.ranking_data import build_train_ranking_examples, load_ranking_examples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--train-csv", type=Path, default=ROOT / "train.csv")
    p.add_argument(
        "--model",
        type=Path,
        default=ROOT / "llama31-1b-movielens-full-final",
        help="Frozen LLM: HF id or local PEFT folder",
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rec-lambda", type=float, default=0.1)
    p.add_argument("--max-train", type=int, default=2000)
    p.add_argument("--n-history", type=int, default=10)
    p.add_argument("--no-injection", action="store_true", help="Text-only baseline finetune")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.max_train = 20

    random.seed(args.seed)
    device = pick_device()
    print(f"[device] {device}")

    train_df = load_interactions(args.train_csv, min_rating=0.0)
    movie_meta = (
        pd.read_csv(args.train_csv, usecols=["MovieID", "Title", "Genres"])
        .drop_duplicates("MovieID")
    )
    id_maps = IdMaps.from_json(args.checkpoint_dir / "id_maps.json")

    examples = build_train_ranking_examples(
        train_df,
        id_maps,
        movie_meta,
        n_history=args.n_history,
        max_examples=args.max_train,
        seed=args.seed,
    )
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    print(f"[data] {len(examples)} training ranking examples")

    model = InjectedLlamaRanker(
        model_name=args.model,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        freeze_llm=True,
        train_adapter=not args.no_injection,
    )
    if args.no_injection:
        for p in model.adapter.parameters():
            p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = __import__("torch").optim.AdamW(trainable, lr=args.lr)

    use_inj = not args.no_injection
    for epoch in range(1, args.epochs + 1):
        model.adapter.train()
        model.llm.eval()
        running = 0.0
        t0 = time.time()
        for ex in tqdm(examples, desc=f"epoch {epoch}"):
            optimizer.zero_grad()
            loss, _ = model.forward_batch(
                [ex],
                [ex.true_position],
                use_injection=use_inj,
                rec_lambda=args.rec_lambda if use_inj else 0.0,
            )
            if loss is None:
                continue
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu())

        avg = running / max(len(examples), 1)
        print(f"epoch {epoch}/{args.epochs}  loss={avg:.4f}  ({time.time()-t0:.1f}s)")

    out = args.checkpoint_dir / "adapter_llm.pt"
    import torch

    torch.save(
        {
            "adapter_state_dict": model.adapter.state_dict(),
            "config": {
                "model": args.model,
                "rec_lambda": args.rec_lambda,
                "use_injection": use_inj,
                "n_history": args.n_history,
            },
        },
        out,
    )
    print(f"[done] saved {out}")


if __name__ == "__main__":
    main()
