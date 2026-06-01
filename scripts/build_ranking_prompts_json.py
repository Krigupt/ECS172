#!/usr/bin/env python3
"""Build ranking prompt JSON for LLM4Rec (history + candidates + injection fields).

Output format matches test_ranking_prompts.json (not train_prompts.json).

train_prompts.json  = SFT for *rating prediction* (one target movie, completion = 1-5 stars).
This script         = *candidate ranking* (N candidates, completion = position 1..N).

Example:
    python scripts/build_ranking_prompts_json.py --split train --n-candidates 5
    python scripts/build_ranking_prompts_json.py --split test --max-samples 100 --smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data import IdMaps, load_interactions
from src.ranking_data import INJECT_SPLIT, build_train_ranking_examples, load_ranking_examples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--train-csv", type=Path, default=ROOT / "train.csv")
    p.add_argument("--test-json", type=Path, default=ROOT / "test_ranking_prompts.json")
    p.add_argument("--test-csv", type=Path, default=ROOT / "test_ranking.csv")
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--n-history", type=int, default=10)
    p.add_argument("--n-candidates", type=int, default=5)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--smoke", action="store_true", help="Write 50 rows only")
    return p.parse_args()


def example_to_json_row(ex, meta_index) -> dict:
    """One record aligned with test_ranking_prompts.json + SFT completion field."""
    tid = ex.true_positive_movie_id
    title = ""
    if tid in meta_index.index:
        row = meta_index.loc[tid]
        title = str(row["Title"] if "Title" in row else row.title)

    return {
        "UserID": ex.user_id,
        "prompt": ex.prompt,
        "prefix_text": ex.prefix_text,
        "suffix_text": ex.suffix_text,
        "inject_split": INJECT_SPLIT,
        "history_movie_ids": ex.history_movie_ids,
        "candidates": ex.candidate_movie_ids,
        "true_positive_id": tid,
        "true_positive_Title": title,
        "true_positive_pos": ex.true_position - 1,  # 0-based index (matches test JSON)
        "true_positive_pos_1indexed": ex.true_position,  # 1..N for your eval
        "completion": str(ex.true_position),
        "n_history": len(ex.history_movie_ids),
        "n_candidates": len(ex.candidate_movie_ids),
    }


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.max_samples = args.max_samples or 50

    id_maps = IdMaps.from_json(args.checkpoint_dir / "id_maps.json")

    if args.split == "test":
        examples = load_ranking_examples(
            args.test_json, args.test_csv, id_maps, n_history=args.n_history
        )
        meta = pd.read_csv(args.train_csv, usecols=["MovieID", "Title", "Genres"]).drop_duplicates(
            "MovieID"
        )
        if args.n_candidates < 10:
            from src.ranking_data import apply_n_candidates

            examples = apply_n_candidates(examples, args.n_candidates, seed=args.seed)
        default_out = ROOT / f"test_ranking_prompts_{args.n_candidates}cand.json"
    else:
        train_df = load_interactions(args.train_csv, min_rating=0.0)
        meta = pd.read_csv(args.train_csv, usecols=["MovieID", "Title", "Genres"]).drop_duplicates(
            "MovieID"
        )
        examples = build_train_ranking_examples(
            train_df,
            id_maps,
            meta,
            n_history=args.n_history,
            n_candidates=args.n_candidates,
            max_examples=args.max_samples,
            seed=args.seed,
        )
        default_out = ROOT / f"train_ranking_prompts_{args.n_candidates}cand.json"

    if args.max_samples and args.split == "test":
        examples = examples[: args.max_samples]

    meta_index = meta.set_index("MovieID")
    rows = [example_to_json_row(ex, meta_index) for ex in examples]

    out = args.output or default_out
    out.write_text(json.dumps(rows, indent=2))
    print(f"[done] wrote {len(rows)} rows -> {out}")
    print(f"  split={args.split}  history={args.n_history}  candidates={args.n_candidates}")
    print(f"  inject marker: {INJECT_SPLIT!r}")
    if rows:
        print(f"  sample completion (answer): {rows[0]['completion']!r}")


if __name__ == "__main__":
    main()
