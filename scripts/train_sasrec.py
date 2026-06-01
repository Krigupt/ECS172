#!/usr/bin/env python3
"""Train SASRec matching Kang & McAuley (2018).

- Implicit feedback (rating >= 4)
- BCE + 1 negative per timestep
- max_len=200 for MovieLens-1M
- Eval: rank 1 positive + 100 negatives (Hit@10, NDCG@10)

Example:
    python scripts/train_sasrec.py --smoke
    python scripts/train_sasrec.py --epochs 15
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import (
    SASRecTrainDataset,
    build_id_maps,
    build_user_sequences,
    build_val_ranking_samples,
    collate_train_batch,
    load_interactions,
    pick_device,
)
from src.sasrec import SASRec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-csv", type=Path, default=ROOT / "train.csv")
    p.add_argument("--val-csv", type=Path, default=ROOT / "val.csv")
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--embed-dim", type=int, default=50)
    p.add_argument("--max-len", type=int, default=200)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--min-rating", type=float, default=4.0)
    p.add_argument("--num-negatives", type=int, default=100, help="Val negatives per user (paper)")
    p.add_argument("--max-users", type=int, default=None)
    p.add_argument("--val-limit", type=int, default=None)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def evaluate_ranking(
    model: SASRec,
    val_samples: list,
    device: torch.device,
    batch_size: int = 64,
) -> dict:
    if not val_samples:
        return {"val_loss": float("nan"), "val_hr@10": float("nan"), "val_ndcg@10": float("nan")}

    model.eval()
    hits = 0
    ndcg_sum = 0.0
    n = 0

    for start in range(0, len(val_samples), batch_size):
        batch = val_samples[start : start + batch_size]
        inputs = torch.stack([s.input_ids for s in batch]).to(device)
        candidates = torch.stack([s.candidate_ids for s in batch]).to(device)
        targets = torch.tensor([s.target_id for s in batch], device=device)

        scores = model.score_candidates(inputs, candidates)
        top10_idx = scores.topk(min(10, scores.size(1)), dim=1).indices

        for i in range(len(batch)):
            ranked_items = candidates[i, top10_idx[i]].tolist()
            t = targets[i].item()
            if t in ranked_items:
                hits += 1
                rank = ranked_items.index(t) + 1
                ndcg_sum += 1.0 / math.log2(rank + 1)
            n += 1

    return {"val_hr@10": hits / n, "val_ndcg@10": ndcg_sum / n, "n_val": n}


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.max_users = args.max_users or 300
        args.epochs = min(args.epochs, 3)
        args.batch_size = min(args.batch_size, 64)
        args.max_len = min(args.max_len, 50)
        args.val_limit = args.val_limit or 200

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = pick_device()
    ckpt_dir = args.checkpoint_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"[device] {device}")
    print(f"[data] loading {args.train_csv} (implicit: rating >= {args.min_rating})")
    train_df = load_interactions(args.train_csv, min_rating=args.min_rating)
    val_df = load_interactions(args.val_csv, min_rating=args.min_rating)

    if args.max_users:
        all_users = train_df["UserID"].unique().tolist()
        picked = set(rng.sample(all_users, min(args.max_users, len(all_users))))
        train_df = train_df[train_df["UserID"].isin(picked)]

    val_movies = val_df["MovieID"].astype(int).unique().tolist()
    id_maps = build_id_maps(train_df, extra_movie_ids=val_movies)
    id_maps.to_json(ckpt_dir / "id_maps.json")
    print(f"[data] n_items={id_maps.n_items}, users={train_df['UserID'].nunique()}")

    sequences, seen_sets = build_user_sequences(train_df, id_maps)
    print(f"[data] {len(sequences)} training sequences")

    train_ds = SASRecTrainDataset(sequences, seen_sets, max_len=args.max_len)
    collate_fn = partial(collate_train_batch, n_items=id_maps.n_items, rng=rng)
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    val_samples = build_val_ranking_samples(
        train_df,
        val_df,
        id_maps,
        max_len=args.max_len,
        num_negatives=args.num_negatives,
        seed=args.seed,
    )
    if args.val_limit:
        val_samples = val_samples[: args.val_limit]
    print(f"[data] {len(val_samples)} val ranking samples (1 pos + {args.num_negatives} negs)")

    model = SASRec(
        n_items=id_maps.n_items,
        embed_dim=args.embed_dim,
        max_len=args.max_len,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(
        f"[train] SASRec paper protocol | epochs={args.epochs} | batch={args.batch_size} | "
        f"max_len={args.max_len} | d={args.embed_dim} | BCE+1neg"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for input_ids, targets, neg_ids, valid_mask in tqdm(loader, desc=f"epoch {epoch}", leave=False):
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            neg_ids = neg_ids.to(device)
            valid_mask = valid_mask.to(device)

            optimizer.zero_grad()
            loss = model.bce_loss(input_ids, targets, neg_ids, valid_mask)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        metrics = evaluate_ranking(model, val_samples, device)
        elapsed = time.time() - t0
        print(
            f"epoch {epoch}/{args.epochs}  "
            f"train_bce={train_loss:.4f}  "
            f"HR@10={metrics['val_hr@10']:.4f}  "
            f"NDCG@10={metrics['val_ndcg@10']:.4f}  "
            f"(n_val={metrics['n_val']}, {elapsed:.1f}s)"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "n_items": id_maps.n_items,
                "embed_dim": args.embed_dim,
                "max_len": args.max_len,
                "n_layers": args.n_layers,
                "dropout": args.dropout,
                "min_rating": args.min_rating,
                "protocol": "sasrec_paper_bce",
            },
        },
        ckpt_dir / "sasrec.pt",
    )
    torch.save(model.item_embedding_matrix(), ckpt_dir / "item_embeddings.pt")
    print(f"[done] saved {ckpt_dir / 'sasrec.pt'}")
    print(f"[done] saved {ckpt_dir / 'item_embeddings.pt'}  shape={model.item_embedding_matrix().shape}")


if __name__ == "__main__":
    main()
