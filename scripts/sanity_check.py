#!/usr/bin/env python3
"""Pre-flight checks before running the A/C eval. Run in Colab after caching.

Verifies the two places where the pipeline fails silently (wrong numbers, no crash):
  1. movie_id -> soft-token cache round-trips with the right shape, and the
     id_maps indexing lines up with projected_embeddings.
  2. Each candidate letter ' A'..' J' is a SINGLE token for this tokenizer, so
     the log-prob scoring reads one logit per letter.

Example:
    python scripts/sanity_check.py --model unsloth/Llama-3.2-1B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import IdMaps

LETTERS = "ABCDEFGHIJ"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--json", type=Path, default=ROOT / "test_ranking_prompts.json")
    p.add_argument("--model", type=str, default="unsloth/Llama-3.2-1B-Instruct")
    p.add_argument("--n-letters", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ok = True

    # ── 1. Cache round-trip ───────────────────────────────────────────
    print("=" * 60)
    print("CHECK 1: movie_id -> soft-token cache")
    print("=" * 60)
    id_maps = IdMaps.from_json(args.checkpoint_dir / "id_maps.json")
    item_emb = torch.load(args.checkpoint_dir / "item_embeddings.pt", map_location="cpu", weights_only=True)
    proj = torch.load(args.checkpoint_dir / "projected_embeddings.pt", map_location="cpu", weights_only=True)
    print(f"  n_items (id_maps)       = {id_maps.n_items}")
    print(f"  item_embeddings shape   = {tuple(item_emb.shape)}")
    print(f"  projected_embeddings    = {tuple(proj.shape)}")

    if proj.shape[0] != item_emb.shape[0]:
        print(f"  [FAIL] row count mismatch: proj {proj.shape[0]} vs items {item_emb.shape[0]}")
        ok = False
    else:
        print(f"  [ok] row counts match ({proj.shape[0]})")

    # Round-trip a real candidate movie id that IS in the vocab.
    rows = json.loads(args.json.read_text())
    all_cands = [int(c) for r in rows for c in r["candidates"]]
    in_vocab = next((m for m in all_cands if m in id_maps.movie_to_idx), None)
    if in_vocab is None:
        print("  [FAIL] no test candidate is in id_maps — vocab mismatch")
        ok = False
    else:
        idx = id_maps.movie_to_idx[in_vocab]
        vec = proj[idx - 1]  # id_maps is 1-based, tensor is 0-based
        print(f"  round-trip movie {in_vocab} -> idx {idx} -> vec shape {tuple(vec.shape)}")
        if vec.ndim != 1:
            print(f"  [FAIL] expected 1-D soft token, got {vec.ndim}-D")
            ok = False
        else:
            print(f"  [ok] soft token dim = {vec.shape[0]}")

    # Report OOV coverage (informational — some test movies are outside SASRec's
    # vocab and get zero vectors in Mode C; small rates are expected, not a failure).
    uniq = set(all_cands)
    oov = {m for m in uniq if m not in id_maps.movie_to_idx}
    true_oov = sum(1 for r in rows if int(r["true_positive_id"]) not in id_maps.movie_to_idx)
    print(
        f"  [info] OOV candidates: {len(oov)}/{len(uniq)} unique "
        f"({100*len(oov)/len(uniq):.1f}%); true item OOV in {true_oov}/{len(rows)} examples"
    )
    if true_oov > 0.05 * len(rows):
        print("  [warn] >5% of examples have an OOV true item — Mode C is materially handicapped")

    # ── 2. Letter tokenization ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("CHECK 2: single-token letters")
    print("=" * 60)
    tok = AutoTokenizer.from_pretrained(args.model)
    for ch in LETTERS[: args.n_letters]:
        ids = tok.encode(f" {ch}", add_special_tokens=False)
        status = "ok" if len(ids) == 1 else "FAIL"
        if len(ids) != 1:
            ok = False
        print(f"  ' {ch}' -> {ids}  [{status}]")

    print("\n" + "=" * 60)
    print("RESULT:", "ALL CHECKS PASSED ✅" if ok else "CHECKS FAILED ❌")
    print("=" * 60)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
