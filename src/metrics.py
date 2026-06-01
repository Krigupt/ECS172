"""Top-K ranking metrics (Hit Rate, NDCG)."""

from __future__ import annotations

import math
from typing import Iterable, List


def hit_rate_at_k(ranking: List[int], target: int, k: int) -> float:
    return 1.0 if target in ranking[:k] else 0.0


def ndcg_at_k(ranking: List[int], target: int, k: int) -> float:
    top_k = ranking[:k]
    if target not in top_k:
        return 0.0
    rank = top_k.index(target) + 1
    return 1.0 / math.log2(rank + 1)


def evaluate_position_predictions(
    predicted_positions: Iterable[int],
    true_positions: Iterable[int],
    ks: Iterable[int] = (1, 5, 10),
) -> dict:
    """Single predicted index per user (1..10). HR@K = accuracy when only top-1 is predicted."""
    ks = list(ks)
    preds = list(predicted_positions)
    trues = list(true_positions)
    n = len(preds)
    if n == 0:
        return {"n": 0}
    hits = sum(1 for p, t in zip(preds, trues) if p > 0 and p == t)
    ndcg_hit = sum((1.0 / math.log2(2)) if p > 0 and p == t else 0.0 for p, t in zip(preds, trues)) / n
    out = {"n": n, "accuracy": hits / n}
    for k in ks:
        out[f"HR@{k}"] = hits / n
        out[f"NDCG@{k}"] = ndcg_hit
    return out
