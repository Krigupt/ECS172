#!/usr/bin/env python3
"""Phase 3: Letter-based (A-J) log-prob ranking eval — Mode A vs Mode C.

Two conditions only (B/D dropped per project scope):
    Mode A — text baseline: candidates listed as titles (A. Title ... J. Title).
    Mode C — candidates as soft tokens: history stays text, each candidate line
             is a letter label + the cached adapter soft token for that movie.

Both score by the log-prob the model assigns to each letter token (' A'..' J')
at the answer position, then rank candidates by that probability. This gives a
full ranked list, so HR@K / NDCG@K genuinely diverge across K.

The LLM is loaded once; all requested modes run against it.

Example:
    python scripts/eval_ranking.py --modes A C
    python scripts/eval_ranking.py --modes C --max-samples 200 --n-candidates 5
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
from src.metrics import evaluate_ranked_lists, random_baseline
from src.ranking_data import apply_n_candidates, load_ranking_examples

LETTERS = "ABCDEFGHIJ"
MODE_TO_INTERNAL = {"A": "text", "C": "candidates"}


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
        "--modes",
        nargs="+",
        choices=["A", "C"],
        default=["A", "C"],
        help="Which conditions to run. A=text baseline, C=candidate soft tokens.",
    )
    p.add_argument(
        "--embedding-adapter",
        type=Path,
        default=None,
        help="Override path to embedding adapter checkpoint (default checkpoints/adapter.pt). Mode C only.",
    )
    p.add_argument(
        "--projected-embeddings",
        type=Path,
        default=None,
        help="Override path to projected_embeddings.pt cache. Use projected_embeddings_ranking.pt "
             "when evaluating the ranking-trained adapter. Default: checkpoints/projected_embeddings.pt",
    )
    p.add_argument(
        "--chat-template",
        action="store_true",
        help="Wrap prompts in the chat template. Default off (raw 'Answer:' cue, "
        "which makes the ' A'..' J' single-token scoring deterministic).",
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--n-history", type=int, default=10)
    p.add_argument(
        "--n-candidates",
        type=int,
        default=10,
        help="Candidates per user (subsampled from the 10 in the test set if < 10). "
        "HR@K/NDCG@K are only reported for K < n_candidates (K==n is trivially 1.0).",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output-dir", type=Path, default=ROOT)
    return p.parse_args()


def count_oov(model, examples):
    """How many candidates have no soft token (fall back to zero vector) in Mode C."""
    vocab = model.id_maps.movie_to_idx
    total = sum(len(ex.candidate_movie_ids) for ex in examples)
    oov = sum(1 for ex in examples for mid in ex.candidate_movie_ids if int(mid) not in vocab)
    true_oov = sum(1 for ex in examples if int(ex.true_positive_movie_id) not in vocab)
    return oov, total, true_oov


def run_mode(model, examples, mode_letter, chat_template, ks, n_candidates):
    """Run one condition; return (metrics, predictions list)."""
    internal = MODE_TO_INTERNAL[mode_letter]
    ranked_lists = []
    true_indices = []
    predictions = []
    t0 = time.time()

    for i, ex in enumerate(examples):
        ranked, probs = model.rank_by_logprob(
            ex, mode=internal, use_chat_template=chat_template
        )
        true_idx = ex.true_position - 1  # true_position is 1-based; convert to 0-based
        ranked_lists.append(ranked)
        true_indices.append(true_idx)

        pred_idx = ranked[0]
        predictions.append(
            {
                "user_id": ex.user_id,
                "pred_letter": LETTERS[pred_idx],
                "true_letter": LETTERS[true_idx] if 0 <= true_idx < len(LETTERS) else "?",
                "correct": pred_idx == true_idx,
                "ranked_letters": [LETTERS[j] for j in ranked],
                "probs": {LETTERS[j]: round(probs[j], 4) for j in range(len(probs))},
            }
        )
        if i < 3:
            print(
                f"  sample user {ex.user_id}: pred={LETTERS[pred_idx]} "
                f"true={LETTERS[true_idx]} P={probs[pred_idx]:.3f}",
                flush=True,
            )
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(examples)}] running...", flush=True)

    metrics = evaluate_ranked_lists(ranked_lists, true_indices, ks=ks)
    metrics["seconds"] = round(time.time() - t0, 1)
    return metrics, predictions


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
    print(f"[setup] modes={args.modes} candidates={args.n_candidates} chat_template={args.chat_template}", flush=True)
    print(f"[llm] {args.model}", flush=True)
    print("[load] loading LLM (first run may download base weights)...", flush=True)

    need_inject = "C" in args.modes
    emb_path = args.embedding_adapter or (args.checkpoint_dir / "adapter.pt")

    # Warn if the adapter was trained against a different LLM than the one we're evaluating with.
    # Soft tokens are only meaningful relative to the exact model they were trained against.
    if need_inject and emb_path.exists():
        _ckpt_cfg = torch.load(emb_path, map_location="cpu", weights_only=False).get("config", {})
        _trained_on = _ckpt_cfg.get("llm_model")
        _eval_model = str(args.model)
        _method = _ckpt_cfg.get("training_method", "unknown")
        print(f"[adapter] {emb_path.name}  method={_method}", flush=True)
        if _trained_on:
            print(f"[adapter] trained against: {_trained_on}", flush=True)
            if _trained_on not in _eval_model and _eval_model not in _trained_on:
                print(
                    f"[WARN] adapter trained on '{_trained_on}' but eval uses '{_eval_model}'. "
                    "Soft tokens may not transfer — results could be meaningless.",
                    flush=True,
                )
        if _ckpt_cfg.get("best_val_hr1") is not None:
            print(f"[adapter] best_val_HR@1={_ckpt_cfg['best_val_hr1']:.4f}", flush=True)

    proj_emb_path = args.projected_embeddings
    if proj_emb_path is None:
        proj_emb_path = args.checkpoint_dir / "projected_embeddings.pt"
    print(f"[cache] projected_embeddings: {proj_emb_path.name}", flush=True)

    model = InjectedLlamaRanker(
        model_name=args.model,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        freeze_llm=True,
        train_adapter=False,
        load_embedding_adapter=need_inject,
        embedding_adapter_path=emb_path if need_inject else None,
        projected_embeddings_path=proj_emb_path,
    )

    # K == n_candidates is trivially 1.0 (true item is always within the full list),
    # so only report cutoffs strictly below the candidate count.
    ks = [k for k in (1, 3, 5, 10) if k < args.n_candidates] or [1]
    rand = random_baseline(len(examples), n_candidates=args.n_candidates, ks=ks, seed=args.seed)

    if "C" in args.modes:
        oov, total, true_oov = count_oov(model, examples)
        print(
            f"[coverage] Mode C: {oov}/{total} candidate slots have no soft token "
            f"(zero vector) = {100*oov/max(total,1):.1f}%; true item OOV in {true_oov} examples",
            flush=True,
        )

    summary = {"random": {k: round(v, 4) for k, v in rand.items()}}

    for mode_letter in args.modes:
        label = "Mode A (text baseline)" if mode_letter == "A" else "Mode C (candidate soft tokens)"
        print("\n" + "=" * 60, flush=True)
        print(label, flush=True)
        print("=" * 60, flush=True)
        metrics, predictions = run_mode(
            model, examples, mode_letter, args.chat_template, ks, args.n_candidates
        )
        summary[f"Mode {mode_letter}"] = {k: round(v, 4) for k, v in metrics.items()}

        print(f"\n=== Results: {label} ===", flush=True)
        for k, v in metrics.items():
            print(f"  {k}: {v}")

        out_path = args.output_dir / f"results_mode_{mode_letter}.json"
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        out_path.write_text(
            json.dumps(
                {
                    "config": config,
                    "mode": mode_letter,
                    "metrics": metrics,
                    "random_baseline": rand,
                    "predictions": predictions,
                },
                indent=2,
            )
        )
        print(f"[done] wrote {out_path}", flush=True)

    # ── Side-by-side summary ──────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    hdr = f"{'Condition':<32}" + "".join(f"{f'HR@{k}':>8}" for k in ks) + f"{'MRR':>8}"
    print(hdr)
    print("-" * len(hdr))
    for name, m in summary.items():
        row = (
            f"{name:<32}"
            + "".join(f"{m.get(f'HR@{k}', 0):>8.3f}" for k in ks)
            + f"{m.get('MRR', 0):>8.3f}"
        )
        print(row)


if __name__ == "__main__":
    main()
