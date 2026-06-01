"""MovieLens loaders and SASRec datasets (paper-faithful protocol)."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


PAD_ID = 0


@dataclass
class IdMaps:
    movie_to_idx: Dict[int, int]
    idx_to_movie: Dict[int, int]
    n_items: int

    def to_json(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "movie_to_idx": {str(int(k)): int(v) for k, v in self.movie_to_idx.items()},
                    "idx_to_movie": {str(int(k)): int(v) for k, v in self.idx_to_movie.items()},
                    "n_items": int(self.n_items),
                },
                indent=2,
            )
        )

    @classmethod
    def from_json(cls, path: Path) -> "IdMaps":
        raw = json.loads(path.read_text())
        movie_to_idx = {int(k): int(v) for k, v in raw["movie_to_idx"].items()}
        idx_to_movie = {int(k): int(v) for k, v in raw["idx_to_movie"].items()}
        return cls(movie_to_idx=movie_to_idx, idx_to_movie=idx_to_movie, n_items=raw["n_items"])


def load_interactions(
    csv_path: str | Path,
    min_rating: float = 4.0,
) -> pd.DataFrame:
    """Implicit feedback: keep interactions with rating >= min_rating."""
    df = pd.read_csv(csv_path, usecols=["UserID", "MovieID", "Rating", "Timestamp"])
    df = df[df["Rating"] >= min_rating]
    df.sort_values(["UserID", "Timestamp"], inplace=True)
    return df


def build_id_maps(
    train_df: pd.DataFrame,
    extra_movie_ids: Optional[List[int]] = None,
) -> IdMaps:
    movies = set(train_df["MovieID"].astype(int).unique())
    if extra_movie_ids:
        movies.update(int(m) for m in extra_movie_ids)
    sorted_movies = sorted(movies)
    movie_to_idx = {mid: i + 1 for i, mid in enumerate(sorted_movies)}
    idx_to_movie = {i + 1: mid for i, mid in enumerate(sorted_movies)}
    return IdMaps(movie_to_idx=movie_to_idx, idx_to_movie=idx_to_movie, n_items=len(sorted_movies))


def build_user_sequences(
    df: pd.DataFrame,
    id_maps: IdMaps,
    min_len: int = 2,
) -> Tuple[List[List[int]], List[Set[int]]]:
    """Chronological item-index sequences + per-user seen sets (for negative sampling)."""
    sequences: List[List[int]] = []
    seen_sets: List[Set[int]] = []
    for _, group in df.groupby("UserID", sort=False):
        mids = group["MovieID"].astype(int).tolist()
        seq = [id_maps.movie_to_idx[m] for m in mids if m in id_maps.movie_to_idx]
        if len(seq) >= min_len:
            sequences.append(seq)
            seen_sets.append(set(seq))
    return sequences, seen_sets


def sequence_to_tensors(
    seq: List[int],
    max_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Paper: input (s1..s_{n-1}), shifted output (s2..s_n), left-padded to max_len."""
    if len(seq) < 2:
        raise ValueError("sequence too short")

    inp = seq[:-1]
    tgt = seq[1:]
    if len(inp) > max_len:
        inp = inp[-max_len:]
        tgt = tgt[-max_len:]

    pad = max_len - len(inp)
    padded_in = [PAD_ID] * pad + inp
    padded_tgt = [PAD_ID] * pad + tgt
    valid = torch.tensor(
        [1 if (padded_in[t] != PAD_ID and padded_tgt[t] != PAD_ID) else 0 for t in range(max_len)],
        dtype=torch.bool,
    )
    return (
        torch.tensor(padded_in, dtype=torch.long),
        torch.tensor(padded_tgt, dtype=torch.long),
        valid,
    )


class SASRecTrainDataset(Dataset):
    """One sample per user: full padded sequence with all timestep targets."""

    def __init__(
        self,
        sequences: List[List[int]],
        seen_sets: List[Set[int]],
        max_len: int = 200,
    ):
        self.samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Set[int]]] = []
        for seq, seen in zip(sequences, seen_sets):
            try:
                inp, tgt, valid = sequence_to_tensors(seq, max_len)
            except ValueError:
                continue
            if valid.any():
                self.samples.append((inp, tgt, valid, seen))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        inp, tgt, valid, seen = self.samples[idx]
        return inp, tgt, valid, seen


def collate_train_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Set[int]]],
    n_items: int,
    rng: random.Random,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stack batch and sample one negative per (B, L) from items not in user history."""
    inputs = torch.stack([b[0] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    valid = torch.stack([b[2] for b in batch])

    neg = torch.zeros_like(targets)
    all_items = list(range(1, n_items + 1))

    for b, (_, _, _, seen) in enumerate(batch):
        seen_set = seen
        pool = [i for i in all_items if i not in seen_set]
        if not pool:
            pool = all_items
        for t in range(targets.size(1)):
            if valid[b, t]:
                neg[b, t] = rng.choice(pool)

    return inputs, targets, neg, valid


@dataclass
class RankingEvalSample:
    input_ids: torch.Tensor
    candidate_ids: torch.Tensor
    target_id: int


def build_val_ranking_samples(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    id_maps: IdMaps,
    max_len: int = 200,
    num_negatives: int = 100,
    seed: int = 42,
) -> List[RankingEvalSample]:
    """Paper eval: rank 1 ground-truth + num_negatives random negatives."""
    rng = random.Random(seed)
    train_by_user: Dict[int, List[int]] = {}
    seen_by_user: Dict[int, Set[int]] = {}
    for uid, group in train_df.groupby("UserID", sort=False):
        seq = [
            id_maps.movie_to_idx[int(m)]
            for m in group["MovieID"]
            if int(m) in id_maps.movie_to_idx
        ]
        train_by_user[int(uid)] = seq
        seen_by_user[int(uid)] = set(seq)

    all_items = list(range(1, id_maps.n_items + 1))
    samples: List[RankingEvalSample] = []

    for row in val_df.itertuples(index=False):
        uid = int(row.UserID)
        mid = int(row.MovieID)
        if mid not in id_maps.movie_to_idx:
            continue
        history = train_by_user.get(uid, [])
        if len(history) < 1:
            continue

        target = id_maps.movie_to_idx[mid]
        seen = seen_by_user.get(uid, set())
        neg_pool = [i for i in all_items if i not in seen and i != target]
        if len(neg_pool) < num_negatives:
            continue
        negatives = rng.sample(neg_pool, num_negatives)
        candidates = [target] + negatives
        rng.shuffle(candidates)

        # Encoder input = train history only; target is ranked among candidates
        inp, _, _ = sequence_to_tensors(history, max_len)

        samples.append(
            RankingEvalSample(
                input_ids=inp,
                candidate_ids=torch.tensor(candidates, dtype=torch.long),
                target_id=target,
            )
        )

    return samples


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
