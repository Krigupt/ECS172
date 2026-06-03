#!/usr/bin/env python3
"""Phase 2b: Train adapter with listwise ranking CE loss (Mode C alignment).

Replaces candidate text lines with adapter soft tokens (Mode C style), evaluates
the frozen LLM's log-probability over A-J letter labels, and backpropagates
cross-entropy to push the adapter toward soft tokens that correctly rank the
true candidate.

Optionally adds a reconstruction grounding term (--recon-lambda > 0) that
teacher-forces the true candidate's title from its soft token. This prevents
the tokens from drifting to semantically empty directions during ranking training.

Training is validated each epoch on a held-out split using the *literal eval
metric* (HR@1 via rank_by_logprob), and the best-HR@1 checkpoint is saved.

Example:
    python scripts/train_adapter_ranking.py --smoke
    python scripts/train_adapter_ranking.py --epochs 3 --max-train 4000 --recon-lambda 0.3
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import IdMaps, load_interactions, pick_device
from src.inject_llm import InjectedLlamaRanker
from src.metrics import evaluate_ranked_lists
from src.ranking_data import build_train_ranking_examples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--train-csv", type=Path, default=ROOT / "train.csv")
    p.add_argument(
        "--model",
        type=str,
        default="unsloth/Llama-3.2-1B-Instruct",
        help="Frozen LLM: HF id or local PEFT folder. Must match the model used at eval time.",
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=4, help="Gradient accumulation steps")
    p.add_argument("--max-train", type=int, default=4000, help="Max training ranking examples")
    p.add_argument("--val-fraction", type=float, default=0.1, help="Fraction held out for val HR@1")
    p.add_argument("--n-history", type=int, default=10)
    p.add_argument("--n-candidates", type=int, default=10, help="Candidates per ranking example")
    p.add_argument(
        "--recon-lambda",
        type=float,
        default=0.3,
        help="Weight for reconstruction grounding loss. 0.0 = pure ranking CE.",
    )
    p.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="Adapter checkpoint to warm-start from (default: checkpoints/adapter.pt "
             "— the reconstruction adapter, loaded automatically by InjectedLlamaRanker).",
    )
    p.add_argument("--use-chat-template", action="store_true", help="Wrap prompt in chat template")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_movie_texts(train_csv: Path, id_maps: IdMaps) -> dict[int, str]:
    """movie_id -> title string for reconstruction grounding."""
    df = pd.read_csv(train_csv, usecols=["MovieID", "Title"], encoding="utf-8", encoding_errors="replace").drop_duplicates("MovieID")
    meta = df.set_index("MovieID")
    texts: dict[int, str] = {}
    for idx in range(1, id_maps.n_items + 1):
        mid = id_maps.idx_to_movie.get(idx)
        if mid is None or mid not in meta.index:
            continue
        texts[int(mid)] = str(meta.loc[mid, "Title"])
    return texts


@torch.no_grad()
def eval_hr1(model: InjectedLlamaRanker, examples, use_chat_template: bool) -> float:
    """Compute HR@1 on val examples using the current (live) adapter state."""
    model.adapter.eval()
    ranked_lists, true_indices = [], []
    for ex in examples:
        ranked, _ = model.rank_by_logprob(ex, mode="candidates", use_chat_template=use_chat_template)
        ranked_lists.append(ranked)
        true_indices.append(ex.true_position - 1)
    m = evaluate_ranked_lists(ranked_lists, true_indices, ks=[1])
    model.adapter.train()
    return m["HR@1"]


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.max_train = 20

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()
    print(f"[device] {device}")
    print(f"[config] model={args.model}  recon_lambda={args.recon_lambda}  chat_template={args.use_chat_template}")

    # ── Load data ──────────────────────────────────────────────────────
    train_df = load_interactions(args.train_csv, min_rating=0.0)
    movie_meta = (
        pd.read_csv(args.train_csv, usecols=["MovieID", "Title", "Genres"], encoding="utf-8", encoding_errors="replace")
        .drop_duplicates("MovieID")
    )
    id_maps = IdMaps.from_json(args.checkpoint_dir / "id_maps.json")

    movie_texts: dict[int, str] = {}
    if args.recon_lambda > 0:
        movie_texts = build_movie_texts(args.train_csv, id_maps)
        print(f"[data] {len(movie_texts)} movie titles loaded for reconstruction grounding")

    # ── Build ranking examples ─────────────────────────────────────────
    all_examples = build_train_ranking_examples(
        train_df,
        id_maps,
        movie_meta,
        n_history=args.n_history,
        n_candidates=args.n_candidates,
        max_examples=args.max_train,
        seed=args.seed,
    )
    rng = random.Random(args.seed)
    rng.shuffle(all_examples)

    n_val = max(1, int(len(all_examples) * args.val_fraction))
    val_examples = all_examples[:n_val]
    train_examples = all_examples[n_val:]

    print(f"[data] {len(train_examples)} train  /  {len(val_examples)} val ranking examples")

    # ── Load model ─────────────────────────────────────────────────────
    # Warm-start: load reconstruction adapter.pt by default (load_embedding_adapter=True).
    # Override with --init-from to resume from a previous ranking checkpoint.
    emb_path = args.init_from or (args.checkpoint_dir / "adapter.pt")
    if not emb_path.exists():
        print(f"[warn] {emb_path} not found — adapter will be randomly initialised")

    model = InjectedLlamaRanker(
        model_name=args.model,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        freeze_llm=True,
        train_adapter=True,
        load_embedding_adapter=emb_path.exists(),
        embedding_adapter_path=emb_path if emb_path.exists() else None,
    )
    init_label = str(emb_path) if emb_path.exists() else "random init"
    print(f"[adapter] warm-start from: {init_label}")

    optimizer = torch.optim.Adam(model.adapter.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.adapter.parameters() if p.requires_grad)
    print(f"[adapter] {n_params:,} trainable params")
    print(f"[train] epochs={args.epochs}  batch_size={args.batch_size}  lr={args.lr}")
    if args.recon_lambda > 0:
        print(f"[train] multi-task: rank_CE + {args.recon_lambda} * recon_CE(title)")
    else:
        print("[train] pure ranking CE (recon_lambda=0)")

    # ── Training loop ──────────────────────────────────────────────────
    best_hr1 = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.adapter.train()
        model.llm.eval()
        rng.shuffle(train_examples)

        running_loss = 0.0
        n_steps = 0
        accum_loss = 0.0
        accum_count = 0
        t0 = time.time()
        optimizer.zero_grad()

        for i, ex in enumerate(tqdm(train_examples, desc=f"epoch {epoch}")):
            if len(ex.candidate_movie_ids) != args.n_candidates:
                continue

            true_title = movie_texts.get(int(ex.true_positive_movie_id)) if args.recon_lambda > 0 else None

            loss = model.forward_contrastive_loss(
                ex,
                use_chat_template=args.use_chat_template,
                recon_lambda=args.recon_lambda,
                true_title=true_title,
            )

            (loss / args.batch_size).backward()
            accum_loss += loss.item()
            accum_count += 1

            if accum_count >= args.batch_size or (i + 1) == len(train_examples):
                optimizer.step()
                optimizer.zero_grad()
                running_loss += accum_loss
                n_steps += accum_count
                accum_loss = 0.0
                accum_count = 0

        avg_loss = running_loss / max(n_steps, 1)
        train_elapsed = time.time() - t0

        # Val: HR@1 using rank_by_logprob with live adapter — same code path as eval
        print(f"  epoch {epoch} train_loss={avg_loss:.4f} ({train_elapsed:.1f}s) — evaluating val HR@1...", flush=True)
        val_hr1 = eval_hr1(model, val_examples, use_chat_template=args.use_chat_template)

        improved = ""
        if val_hr1 > best_hr1:
            best_hr1 = val_hr1
            best_state = {k: v.cpu().clone() for k, v in model.adapter.state_dict().items()}
            improved = " *best*"

        print(
            f"epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  val_HR@1={val_hr1:.4f}"
            f"  best_HR@1={best_hr1:.4f}{improved}"
        )

    # ── Save best checkpoint ───────────────────────────────────────────
    if best_state is not None:
        model.adapter.load_state_dict(best_state)

    out = args.checkpoint_dir / "adapter_ranking.pt"
    torch.save(
        {
            "model_state_dict": model.adapter.state_dict(),
            "config": {
                "sasrec_dim": model.adapter.config.sasrec_dim,
                "llm_dim": model.adapter.config.llm_dim,
                "hidden_dim": model.adapter.config.hidden_dim,
                "training_method": "contrastive_ranking",
                "llm_model": args.model,
                "recon_lambda": args.recon_lambda,
                "use_chat_template": args.use_chat_template,
                "best_val_hr1": best_hr1,
                "n_candidates": args.n_candidates,
                "n_history": args.n_history,
            },
        },
        out,
    )
    print(f"[done] saved {out}  (best val_HR@1={best_hr1:.4f})")
    print("[next] run: python scripts/cache_projected_embeddings.py --adapter checkpoints/adapter_ranking.pt --out checkpoints/projected_embeddings_ranking.pt")


if __name__ == "__main__":
    main()
