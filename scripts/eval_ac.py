#!/usr/bin/env python3
"""Evaluate Mode A (text) vs Mode C (soft-token candidates) using letter-based log-prob ranking.

Produces a full ranked list per example via log-probability scoring over letter
tokens A–J, enabling meaningful HR@K and NDCG@K that diverge across K.

Example:
    # Smoke test (10 examples)
    python scripts/eval_ac.py --smoke

    # Full run with fine-tuned LLM
    python scripts/eval_ac.py --model ./llama31-1b-movielens-full-final --max-samples 200

    # Only Mode A (text baseline)
    python scripts/eval_ac.py --modes A

    # Enable chat template (validate raw first, then try this)
    python scripts/eval_ac.py --chat-template --max-samples 50
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
from src.ranking_data import load_ranking_examples

LETTERS = "ABCDEFGHIJ"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--json", type=Path, default=ROOT / "test_ranking_prompts.json")
    p.add_argument("--csv", type=Path, default=ROOT / "test_ranking.csv")
    p.add_argument(
        "--model",
        type=Path,
        default=ROOT / DEFAULT_LLM_PATH,
        help="HF model id or local PEFT folder",
    )
    p.add_argument(
        "--embedding-adapter",
        type=Path,
        default=None,
        help="Override path to embedding adapter checkpoint (.pt)",
    )
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--n-history", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true", help="Quick test with 10 examples")
    p.add_argument(
        "--modes",
        type=str,
        default="A,C",
        help="Comma-separated modes to run: A (text), C (soft tokens), or A,C",
    )
    p.add_argument(
        "--chat-template",
        action="store_true",
        help="Wrap prompts in chat template (default: raw prompts for validation)",
    )
    p.add_argument("--output", type=Path, default=ROOT / "results_ac.json")
    return p.parse_args()


def print_metrics_table(all_metrics: dict, ks: list) -> None:
    """Pretty-print a comparison table."""
    header = f"{'Mode':<22}"
    for k in ks:
        header += f"  {'HR@' + str(k):>6}"
    for k in ks:
        header += f"  {'NDCG@' + str(k):>8}"
    print(header)
    print("─" * len(header))

    for label, metrics in all_metrics.items():
        row = f"{label:<22}"
        for k in ks:
            row += f"  {metrics.get(f'HR@{k}', 0):>6.3f}"
        for k in ks:
            row += f"  {metrics.get(f'NDCG@{k}', 0):>8.4f}"
        print(row)


def verify_cache_roundtrip(model: InjectedLlamaRanker, id_maps: IdMaps) -> None:
    """Fix #6: verify a known movie's soft token resolves correctly."""
    # Pick the first movie in the ID map
    test_mid = next(iter(id_maps.movie_to_idx.keys()))
    test_idx = id_maps.movie_to_idx[test_mid]
    vec = model.projected_embeddings[test_idx - 1]  # 0-indexed
    expected_dim = model.hidden_size
    assert vec.shape == (expected_dim,), (
        f"Cache round-trip FAILED: movie {test_mid} → idx {test_idx} "
        f"→ shape {vec.shape}, expected ({expected_dim},)"
    )
    print(
        f"[verify] Cache round-trip OK: movie {test_mid} → idx {test_idx} → "
        f"soft token shape {tuple(vec.shape)}",
        flush=True,
    )


def verify_ground_truth(examples: list) -> int:
    """Fix #4: assert candidates[true_positive_pos] == true_positive_id for all examples."""
    n_checked = 0
    n_failed = 0
    for ex in examples:
        true_idx = ex.true_position - 1  # convert 1-indexed to 0-indexed
        if 0 <= true_idx < len(ex.candidate_movie_ids):
            actual_mid = ex.candidate_movie_ids[true_idx]
            if actual_mid != ex.true_positive_movie_id:
                print(
                    f"  [WARN] user {ex.user_id}: candidates[{true_idx}]={actual_mid} "
                    f"!= true_positive_id={ex.true_positive_movie_id}",
                    flush=True,
                )
                n_failed += 1
            n_checked += 1
    print(
        f"[verify] Ground-truth index check: {n_checked} checked, {n_failed} mismatches",
        flush=True,
    )
    return n_failed


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.max_samples = args.max_samples or 10

    modes = [m.strip().upper() for m in args.modes.split(",")]
    for m in modes:
        if m not in ("A", "C"):
            print(f"[error] Unknown mode '{m}'. Use A, C, or A,C.")
            sys.exit(1)

    device = pick_device()
    print(f"[device] {device}", flush=True)
    print(f"[config] chat_template={args.chat_template}", flush=True)

    # ── Load data ─────────────────────────────────────────────────────
    id_maps = IdMaps.from_json(args.checkpoint_dir / "id_maps.json")
    raw_examples = load_ranking_examples(
        args.json, args.csv, id_maps, n_history=args.n_history
    )
    
    # Filter out examples that contain movies not in our id_maps
    # Otherwise Mode C will use zero-vectors and fail.
    valid_examples = []
    missing_count = 0
    for ex in raw_examples:
        has_missing = False
        for mid in ex.candidate_movie_ids:
            if int(mid) not in id_maps.movie_to_idx:
                has_missing = True
                break
        if has_missing:
            missing_count += 1
        else:
            valid_examples.append(ex)
            
    if missing_count > 0:
        print(f"[WARN] Skipped {missing_count} examples because they contained candidate movies missing from id_maps.", flush=True)

    examples = valid_examples
    if args.max_samples:
        examples = examples[: args.max_samples]
        
    if not examples:
        print("[error] No valid examples left to evaluate!")
        sys.exit(1)
        
    n_candidates = len(examples[0].candidate_movie_ids)
    print(f"[data] {len(examples)} examples, {n_candidates} candidates each", flush=True)

    # Fix #4: verify ground-truth scatter
    n_bad = verify_ground_truth(examples)
    if n_bad > 0:
        print(f"[WARN] {n_bad} examples have mismatched ground truth — results may be unreliable", flush=True)

    # ── Load model ────────────────────────────────────────────────────
    emb_path = args.embedding_adapter or args.checkpoint_dir / "adapter.pt"
    need_adapter = "C" in modes

    print(f"[llm] {args.model}", flush=True)
    print("[load] loading LLM (first run may download base weights)...", flush=True)

    model = InjectedLlamaRanker(
        model_name=args.model,
        checkpoint_dir=args.checkpoint_dir,
        device=device,
        freeze_llm=True,
        train_adapter=False,
        load_embedding_adapter=need_adapter,
        embedding_adapter_path=emb_path if need_adapter else None,
    )

    # Fix #6: verify cache round-trip
    if model.projected_embeddings is not None:
        verify_cache_roundtrip(model, id_maps)

    # ── Verify letter token IDs (Fix #2) ──────────────────────────────
    letter_ids = model._get_letter_token_ids(n_candidates)
    print("[tokens] Letter token IDs: ", end="", flush=True)
    for i, tid in enumerate(letter_ids.tolist()):
        ch = LETTERS[i]
        decoded = model.tokenizer.decode([tid])
        print(f"{ch}={tid}({decoded!r}) ", end="")
    print(flush=True)

    # ── Run evaluation ────────────────────────────────────────────────
    ks = [k for k in (1, 3, 5, 10) if k <= n_candidates]
    all_results = {}
    all_predictions = {}

    for mode_label in modes:
        mode_key = "text" if mode_label == "A" else "candidates"
        mode_name = f"Mode {mode_label} ({'text' if mode_label == 'A' else 'soft tokens'})"
        print(f"\n[eval] Running {mode_name}...", flush=True)

        rankings = []
        true_indices = []
        per_example = []
        t0 = time.time()

        for i, ex in enumerate(examples):
            ranked_idx, probs = model.rank_by_logprob(
                ex, mode=mode_key, use_chat_template=args.chat_template,
            )
            true_idx = ex.true_position - 1  # convert 1-indexed to 0-indexed

            rankings.append(ranked_idx)
            true_indices.append(true_idx)

            top1_letter = LETTERS[ranked_idx[0]]
            true_letter = LETTERS[true_idx]
            per_example.append({
                "user_id": ex.user_id,
                "true_letter": true_letter,
                "true_idx": true_idx,
                "top1_letter": top1_letter,
                "correct_top1": ranked_idx[0] == true_idx,
                "true_rank": ranked_idx.index(true_idx) + 1 if true_idx in ranked_idx else -1,
                "probs": {LETTERS[j]: round(p, 6) for j, p in enumerate(probs)},
            })

            if i < 3:
                prob_str = " ".join(
                    f"{LETTERS[j]}={p:.3f}" for j, p in enumerate(probs)
                )
                print(
                    f"  user {ex.user_id}: top1={top1_letter} true={true_letter} "
                    f"rank={per_example[-1]['true_rank']}  [{prob_str}]",
                    flush=True,
                )
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(examples)}]...", flush=True)

        elapsed = time.time() - t0
        metrics = evaluate_ranked_lists(rankings, true_indices, ks)
        all_results[mode_name] = metrics
        all_predictions[mode_name] = per_example
        print(f"  Done in {elapsed:.1f}s", flush=True)

    # ── Random baseline ───────────────────────────────────────────────
    rand_metrics = random_baseline(
        len(examples), n_candidates=n_candidates, ks=ks, seed=args.seed
    )
    all_results["Random baseline"] = rand_metrics

    # ── Print comparison ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RESULTS COMPARISON")
    print(f"{'='*60}")
    print_metrics_table(all_results, ks)
    print()

    # ── Verification summary ──────────────────────────────────────────
    print("[verify] Checking HR@K divergence across K...")
    for mode_name, metrics in all_results.items():
        if mode_name == "Random baseline":
            continue
        hr_values = [metrics.get(f"HR@{k}", 0) for k in ks]
        if len(set(hr_values)) == 1 and len(ks) > 1:
            print(f"  [WARN] {mode_name}: HR@K identical across K — may indicate collapsed metrics")
        else:
            print(f"  [OK] {mode_name}: HR@K diverges across K ✓")

    # ── Save JSON ─────────────────────────────────────────────────────
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    output_data = {
        "config": config,
        "metrics": {k: v for k, v in all_results.items()},
        "predictions": all_predictions,
    }
    args.output.write_text(json.dumps(output_data, indent=2))
    print(f"[done] wrote {args.output}")


if __name__ == "__main__":
    main()
