#!/usr/bin/env python3
"""Real-time style demo: history movie IDs -> adapter injection -> Llama picks candidate.

Example:
    python scripts/predict_ranking.py --user-id 5
    python scripts/predict_ranking.py --history-movie-ids 3186,1270,1721 --candidates 1,2,3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import IdMaps
from src.inject_llm import DEFAULT_LLM_PATH, InjectedLlamaRanker
from src.ranking_data import INJECT_SPLIT, RankingExample, load_ranking_examples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--model", type=Path, default=ROOT / DEFAULT_LLM_PATH)
    p.add_argument("--use-adapter-llm", action="store_true")
    p.add_argument("--user-id", type=int, default=None, help="Load from test_ranking.csv")
    p.add_argument("--history-movie-ids", type=str, default=None, help="Comma-separated MovieIDs")
    p.add_argument("--n-history", type=int, default=10)
    p.add_argument("--no-injection", action="store_true")
    p.add_argument("--train-csv", type=Path, default=ROOT / "train.csv")
    return p.parse_args()


def _meta_for_movies(train_csv: Path, movie_ids: list[int]) -> dict:
    df = pd.read_csv(train_csv, usecols=["MovieID", "Title", "Genres"])
    meta = df.drop_duplicates("MovieID").set_index("MovieID")
    out = {}
    for mid in movie_ids:
        if mid in meta.index:
            row = meta.loc[mid]
            out[mid] = (str(row["Title"]), str(row["Genres"]))
    return out


def build_example_from_ids(
    history_mids: list[int],
    candidate_mids: list[int],
    ratings: list[int] | None,
    meta: dict,
    id_maps: IdMaps,
) -> RankingExample:
    ratings = ratings or [4] * len(history_mids)
    history_lines = []
    history_idx = []
    for mid, r in zip(history_mids, ratings):
        title, genres = meta.get(mid, (f"Movie {mid}", ""))
        history_lines.append(f"- {title} ({genres}): {r}/5")
        if mid in id_maps.movie_to_idx:
            history_idx.append(id_maps.movie_to_idx[mid])
    cand_lines = []
    for j, mid in enumerate(candidate_mids, start=1):
        title, genres = meta.get(mid, (f"Movie {mid}", ""))
        cand_lines.append(f"{j}. {title} ({genres})")
    prefix = "A user has rated the following movies:\n" + "\n".join(history_lines)
    suffix = (
        INJECT_SPLIT
        + ", rank which movie this user would most likely enjoy:\n"
        + "\n".join(cand_lines)
        + "\n\nReply with just the number of the movie they would rate highest."
    )
    return RankingExample(
        user_id=0,
        prompt=prefix + suffix,
        prefix_text=prefix,
        suffix_text=suffix,
        history_movie_ids=history_mids,
        history_item_indices=history_idx,
        candidate_movie_ids=candidate_mids,
        true_position=1,
        true_positive_movie_id=candidate_mids[0],
    )


def main() -> None:
    args = parse_args()
    id_maps = IdMaps.from_json(args.checkpoint_dir / "id_maps.json")

    if args.user_id is not None:
        examples = load_ranking_examples(
            ROOT / "test_ranking_prompts.json",
            ROOT / "test_ranking.csv",
            id_maps,
            n_history=args.n_history,
        )
        ex = next(e for e in examples if e.user_id == args.user_id)
    else:
        if not args.history_movie_ids:
            raise SystemExit("Provide --user-id or --history-movie-ids")
        history = [int(x) for x in args.history_movie_ids.split(",")]
        meta = _meta_for_movies(args.train_csv, history)
        ex = build_example_from_ids(history, list(meta.keys())[:10] if len(meta) >= 10 else history, None, meta, id_maps)

    import torch

    emb_path = args.checkpoint_dir / ("adapter_llm.pt" if args.use_adapter_llm else "adapter.pt")
    model = InjectedLlamaRanker(
        model_name=args.model,
        checkpoint_dir=args.checkpoint_dir,
        train_adapter=False,
        load_embedding_adapter=not args.no_injection,
        embedding_adapter_path=emb_path if not args.no_injection else None,
    )

    pred, raw = model.predict_position(ex, use_injection=not args.no_injection)
    chosen = ex.candidate_movie_ids[pred - 1] if 1 <= pred <= len(ex.candidate_movie_ids) else None
    print(f"Predicted position: {pred}")
    print(f"Raw output: {raw!r}")
    print(f"MovieID: {chosen}")
    print(f"History movies injected: {len(ex.history_item_indices)} vectors")


if __name__ == "__main__":
    main()
