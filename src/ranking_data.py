"""Ranking examples for LLM candidate selection (Phase 3)."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .data import IdMaps

INJECT_SPLIT = "\n\nFrom the list below"


@dataclass
class RankingExample:
    user_id: int
    prompt: str
    prefix_text: str
    suffix_text: str
    history_movie_ids: List[int]
    history_item_indices: List[int]
    candidate_movie_ids: List[int]
    true_position: int
    true_positive_movie_id: int


def _split_prompt(prompt: str) -> tuple[str, str]:
    if INJECT_SPLIT not in prompt:
        raise ValueError("prompt missing injection split marker")
    prefix, rest = prompt.split(INJECT_SPLIT, 1)
    return prefix, INJECT_SPLIT + rest


def prompt_position_from_label(
    candidates: List[int],
    true_movie_id: int,
    raw_pos: int,
) -> int:
    """
    Map dataset label to prompt position 1..10.
    Labels in JSON/CSV are 0-based indices into `candidates`; the prompt lists 1..10.
    """
    cands = [int(c) for c in candidates]
    tid = int(true_movie_id)
    if tid in cands:
        return cands.index(tid) + 1
    if 1 <= int(raw_pos) <= 10:
        return int(raw_pos)
    if 0 <= int(raw_pos) <= 9:
        return int(raw_pos) + 1
    return int(raw_pos) + 1


def load_ranking_json(path: str | Path, id_maps: IdMaps) -> List[RankingExample]:
    rows = json.loads(Path(path).read_text())
    examples: List[RankingExample] = []
    for row in rows:
        prompt = row["prompt"]
        prefix, suffix = _split_prompt(prompt)
        history_mids = row.get("history_movie_ids")
        if history_mids is None:
            continue
        history_idx = [
            id_maps.movie_to_idx[int(m)]
            for m in history_mids
            if int(m) in id_maps.movie_to_idx
        ]
        examples.append(
            RankingExample(
                user_id=int(row["UserID"]),
                prompt=prompt,
                prefix_text=prefix,
                suffix_text=suffix,
                history_movie_ids=[int(m) for m in history_mids],
                history_item_indices=history_idx,
                candidate_movie_ids=[int(c) for c in row["candidates"]],
                true_position=prompt_position_from_label(
                    row["candidates"],
                    row["true_positive_id"],
                    row["true_positive_pos"],
                ),
                true_positive_movie_id=int(row["true_positive_id"]),
            )
        )
    return examples


def load_ranking_csv(path: str | Path, id_maps: IdMaps, n_history: int = 10) -> List[RankingExample]:
    df = pd.read_csv(path)
    examples: List[RankingExample] = []
    for row in df.itertuples(index=False):
        history_mids = [
            int(getattr(row, f"history_{i}_MovieID")) for i in range(1, n_history + 1)
        ]
        history_lines = []
        for i in range(1, n_history + 1):
            title = getattr(row, f"history_{i}_Title")
            rating = getattr(row, f"history_{i}_rating")
            genres = ""
            history_lines.append(f"- {title}: {int(rating)}/5")
        candidate_lines = []
        candidates = []
        for j in range(1, 11):
            mid = int(getattr(row, f"candidate_{j}_MovieID"))
            title = getattr(row, f"candidate_{j}_Title")
            candidates.append(mid)
            candidate_lines.append(f"{j}. {title}")
        prefix = (
            "A user has rated the following movies:\n"
            + "\n".join(history_lines)
        )
        suffix = (
            INJECT_SPLIT
            + ", rank which movie this user would most likely enjoy:\n"
            + "\n".join(candidate_lines)
            + "\n\nReply with just the number of the movie they would rate highest."
        )
        prompt = prefix + suffix
        history_idx = [
            id_maps.movie_to_idx[m] for m in history_mids if m in id_maps.movie_to_idx
        ]
        examples.append(
            RankingExample(
                user_id=int(row.UserID),
                prompt=prompt,
                prefix_text=prefix,
                suffix_text=suffix,
                history_movie_ids=history_mids,
                history_item_indices=history_idx,
                candidate_movie_ids=candidates,
                true_position=prompt_position_from_label(
                    candidates,
                    row.true_positive_MovieID,
                    row.true_positive_pos,
                ),
                true_positive_movie_id=int(row.true_positive_MovieID),
            )
        )
    return examples


def enrich_json_with_history_from_csv(
    json_path: str | Path,
    csv_path: str | Path,
    n_history: int = 10,
) -> List[dict]:
    """Merge CSV history MovieIDs into prompt JSON records."""
    df = pd.read_csv(csv_path)
    by_user = {int(r.UserID): r for r in df.itertuples(index=False)}
    rows = json.loads(Path(json_path).read_text())
    for row in rows:
        uid = int(row["UserID"])
        if uid not in by_user:
            continue
        r = by_user[uid]
        row["history_movie_ids"] = [
            int(getattr(r, f"history_{i}_MovieID")) for i in range(1, n_history + 1)
        ]
    return rows


def _candidate_title_lines(suffix_text: str) -> List[str]:
    """Extract 'Title (genres)' from numbered lines in the candidate block."""
    titles: List[str] = []
    for line in suffix_text.split("\n"):
        line = line.strip()
        m = re.match(r"^\d+\.\s+(.+)$", line)
        if m:
            titles.append(m.group(1))
    return titles


def resample_to_n_candidates(
    example: RankingExample,
    n_candidates: int,
    rng: random.Random,
) -> RankingExample:
    """Keep true item + sample negatives; rebuild prompt for n_candidates (e.g. 5)."""
    if n_candidates >= len(example.candidate_movie_ids):
        return example

    titles = _candidate_title_lines(example.suffix_text)
    if len(titles) != len(example.candidate_movie_ids):
        titles = [f"Movie {mid}" for mid in example.candidate_movie_ids]

    pairs = list(zip(example.candidate_movie_ids, titles))
    true_id = example.true_positive_movie_id
    true_pair = next((p for p in pairs if p[0] == true_id), pairs[0])
    others = [p for p in pairs if p[0] != true_id]
    if len(others) < n_candidates - 1:
        return example

    chosen = [true_pair] + rng.sample(others, n_candidates - 1)
    rng.shuffle(chosen)
    new_ids = [mid for mid, _ in chosen]
    cand_lines = [f"{j}. {title}" for j, (_, title) in enumerate(chosen, start=1)]
    true_pos = new_ids.index(true_id) + 1

    suffix = (
        INJECT_SPLIT
        + ", rank which movie this user would most likely enjoy:\n"
        + "\n".join(cand_lines)
        + f"\n\nReply with just the number (1-{n_candidates}) of the movie they would rate highest."
    )
    prefix = example.prefix_text
    return RankingExample(
        user_id=example.user_id,
        prompt=prefix + suffix,
        prefix_text=prefix,
        suffix_text=suffix,
        history_movie_ids=example.history_movie_ids,
        history_item_indices=example.history_item_indices,
        candidate_movie_ids=new_ids,
        true_position=true_pos,
        true_positive_movie_id=true_id,
    )


def apply_n_candidates(
    examples: List[RankingExample],
    n_candidates: int,
    seed: int = 42,
) -> List[RankingExample]:
    if n_candidates >= 10:
        return examples
    rng = random.Random(seed)
    return [resample_to_n_candidates(ex, n_candidates, rng) for ex in examples]


def load_ranking_examples(
    json_path: str | Path,
    csv_path: str | Path,
    id_maps: IdMaps,
    n_history: int = 10,
) -> List[RankingExample]:
    enriched = enrich_json_with_history_from_csv(json_path, csv_path, n_history)
    tmp = Path(json_path).parent / "_tmp_enriched_ranking.json"
    tmp.write_text(json.dumps(enriched))
    try:
        return load_ranking_json(tmp, id_maps)
    finally:
        if tmp.exists():
            tmp.unlink()


def build_train_ranking_examples(
    train_df: pd.DataFrame,
    id_maps: IdMaps,
    movie_meta: pd.DataFrame,
    n_history: int = 10,
    n_candidates: int = 10,
    max_examples: Optional[int] = None,
    seed: int = 42,
) -> List[RankingExample]:
    """Build leave-one-out ranking samples from train.csv (for finetuning)."""
    rng = random.Random(seed)
    meta = movie_meta.set_index("MovieID")
    all_items = list(range(1, id_maps.n_items + 1))
    examples: List[RankingExample] = []

    for uid, group in train_df.groupby("UserID", sort=False):
        g = group.sort_values("Timestamp")
        if len(g) < n_history + 1:
            continue
        mids = [int(m) for m in g["MovieID"].tolist()]
        ratings = [float(r) for r in g["Rating"].tolist()]
        target_mid = mids[-1]
        if target_mid not in id_maps.movie_to_idx:
            continue
        history = list(zip(mids[-(n_history + 1) : -1], ratings[-(n_history + 1) : -1]))
        seen = set(mids)
        pool = [
            id_maps.idx_to_movie[i]
            for i in all_items
            if id_maps.idx_to_movie[i] not in seen and id_maps.idx_to_movie[i] in meta.index
        ]
        if len(pool) < n_candidates - 1 or target_mid not in meta.index:
            continue
        negs = rng.sample(pool, n_candidates - 1)
        candidates = negs + [target_mid]
        rng.shuffle(candidates)
        true_pos = candidates.index(target_mid) + 1

        history_lines = []
        history_idx = []
        for mid, rating in history:
            if mid not in meta.index or mid not in id_maps.movie_to_idx:
                continue
            row = meta.loc[mid]
            title = row["Title"] if "Title" in row else row.title
            genres = row.get("Genres", "")
            history_lines.append(f"- {title} ({genres}): {int(rating)}/5")
            history_idx.append(id_maps.movie_to_idx[mid])

        if len(history_lines) < n_history:
            continue

        cand_lines = []
        for j, mid in enumerate(candidates, start=1):
            if mid not in meta.index:
                break
            row = meta.loc[mid]
            title = row["Title"] if "Title" in row else row.title
            genres = row.get("Genres", "")
            cand_lines.append(f"{j}. {title} ({genres})")
        if len(cand_lines) != n_candidates:
            continue

        prefix = "A user has rated the following movies:\n" + "\n".join(history_lines)
        suffix = (
            INJECT_SPLIT
            + ", rank which movie this user would most likely enjoy:\n"
            + "\n".join(cand_lines)
            + "\n\nReply with just the number of the movie they would rate highest."
        )
        examples.append(
            RankingExample(
                user_id=int(uid),
                prompt=prefix + suffix,
                prefix_text=prefix,
                suffix_text=suffix,
                history_movie_ids=[m for m, _ in history],
                history_item_indices=history_idx,
                true_position=true_pos,
                true_positive_movie_id=target_mid,
                candidate_movie_ids=candidates,
            )
        )
        if max_examples and len(examples) >= max_examples:
            break

    return examples
