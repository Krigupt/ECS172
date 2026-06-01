#!/usr/bin/env python3
"""Phase 3: Evaluate ranker on test_ranking (HR@10, NDCG@10).

Uses your fine-tuned local Llama (LoRA) for generation; optional SASRec vector injection.

Example:
    # Your MovieLens-tuned LLM, text-only (no SASRec injection)
    python scripts/eval_ranking.py --no-injection

    # Same LLM + SASRec injection (embedding adapter from checkpoints/)
    python scripts/eval_ranking.py --use-adapter-llm

    python scripts/eval_ranking.py --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import IdMaps, pick_device
from src.inject_llm import DEFAULT_LLM_PATH, InjectedLlamaRanker
from src.metrics import evaluate_position_predictions
from src.ranking_data import apply_n_candidates, load_ranking_examples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--json", type=Path, default=ROOT / "test_ranking_prompts.json")
    p.add_argument("--csv", type=Path, default=ROOT / "test_ranking.csv")
    p.add_argument(
        "--model",
        type=Path,
        default=ROOT / DEFAULT_LLM_PATH,
        help="HF model id or local PEFT folder (e.g. llama31-1b-movielens-full-final)",
    )
    p.add_argument(
        "--use-adapter-llm",
        action="store_true",
        help="Use adapter_llm.pt (ranking-tuned embedding adapter); else adapter.pt",
    )
    p.add_argument(
        "--embedding-adapter",
        type=Path,
        default=None,
        help="Override path to embedding adapter checkpoint (.pt)",
    )
    p.add_argument("--no-injection", action="store_true", help="Text-only (no SASRec vectors)")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--n-history", type=int, default=10)
    p.add_argument(
        "--n-candidates",
        type=int,
        default=10,
        help="Number of candidates per user (subsampled from test set if < 10)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output", type=Path, default=ROOT / "results_ranking.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.max_samples = args.max_samples or 10

    device = pick_device()
    id_maps = IdMaps.from_json(args.checkpoint_dir / "id_maps.json")
    examples = load_ranking_examples(args.json, args.csv, id_maps, n_history=args.n_history)
    if args.n_candidates < 10:
        examples = apply_n_candidates(examples, args.n_candidates, seed=args.seed)
    if args.max_samples:
        examples = examples[: args.max_samples]

    print(f"[device] {device}", flush=True)
    print(f"[data] {len(examples)} test examples", flush=True)
    print(f"[setup] history={args.n_history} candidates={args.n_candidates}", flush=True)
    print(f"[llm] {args.model}", flush=True)
    print("[load] loading LLM (first run may download base weights)...", flush=True)

    emb_path = args.embedding_adapter
    if emb_path is None and args.use_adapter_llm:
        emb_path = args.checkpoint_dir / "adapter_llm.pt"
    elif emb_path is None:
        emb_path = args.checkpoint_dir / "adapter.pt"

    model = InjectedLlamaRanker(
        model_name=args.model,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        freeze_llm=True,
        train_adapter=False,
        load_embedding_adapter=not args.no_injection,
        embedding_adapter_path=emb_path if not args.no_injection else None,
    )
    if not args.no_injection:
        print(f"[embedding adapter] {emb_path}", flush=True)
    print("[eval] running inference...", flush=True)

    use_inj = not args.no_injection
    preds = []
    trues = []
    t0 = time.time()

    for i, ex in enumerate(examples):
        pred, raw = model.predict_position(
            ex, use_injection=use_inj, max_position=args.n_candidates
        )
        preds.append(pred)
        trues.append(ex.true_position)
        if i < 3:
            print(f"  sample user {ex.user_id}: pred={pred} true={ex.true_position} raw={raw!r}", flush=True)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(examples)}] running...")

    ks = [k for k in (1, 3, 5, 10) if k <= args.n_candidates]
    metrics = evaluate_position_predictions(preds, trues, ks=ks)
    elapsed = time.time() - t0
    print(f"\n=== Results ({'injection' if use_inj else 'text-only'}) ===")
    for k, v in metrics.items():
        if k != "n":
            print(f"  {k}: {v:.4f}")
    print(f"  n: {metrics['n']}  time: {elapsed:.1f}s")

    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}

    args.output.write_text(
        json.dumps(
            {
                "config": config,
                "metrics": metrics,
                "predictions": [
                    {
                        "user_id": ex.user_id,
                        "pred": p,
                        "true": t,
                        "correct": p == t and p > 0,
                    }
                    for ex, p, t in zip(examples, preds, trues)
                ],
            },
            indent=2,
        )
    )
    print(f"[done] wrote {args.output}")


if __name__ == "__main__":
    main()
