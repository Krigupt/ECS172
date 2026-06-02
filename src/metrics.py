"""Top-K ranking metrics (Hit Rate, NDCG).

Supports both single-prediction and full ranked-list evaluation.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Iterable, List


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


# ── Ranked-list evaluation (log-prob scoring) ────────────────────────


def evaluate_ranked_lists(
    ranked_lists: List[List[int]],
    true_indices: List[int],
    ks: Iterable[int] = (1, 3, 5, 10),
) -> Dict[str, float]:
    """Evaluate a collection of ranked lists against ground truth.

    Args:
        ranked_lists: each element is a list of candidate indices (0-based)
                      sorted by model preference (best first).
        true_indices: the true candidate index (0-based) for each example.
        ks: cutoff values for HR@K and NDCG@K.

    Returns:
        dict with keys 'n', 'HR@K', 'NDCG@K' for each K.
    """
    ks = list(ks)
    n = len(ranked_lists)
    if n == 0:
        return {"n": 0}

    results: Dict[str, float] = {"n": n}
    for k in ks:
        hr_sum = 0.0
        ndcg_sum = 0.0
        for ranking, target in zip(ranked_lists, true_indices):
            hr_sum += hit_rate_at_k(ranking, target, k)
            ndcg_sum += ndcg_at_k(ranking, target, k)
        results[f"HR@{k}"] = hr_sum / n
        results[f"NDCG@{k}"] = ndcg_sum / n

    # Mean Reciprocal Rank over the full list — the position-aware single number
    # that does NOT degenerate to 1.0 when k reaches the candidate count.
    mrr = 0.0
    for ranking, target in zip(ranked_lists, true_indices):
        if target in ranking:
            mrr += 1.0 / (ranking.index(target) + 1)
    results["MRR"] = mrr / n
    return results


def random_baseline(
    n_examples: int,
    n_candidates: int = 10,
    ks: Iterable[int] = (1, 3, 5, 10),
    seed: int = 42,
) -> Dict[str, float]:
    """Compute expected metrics for a random ranker (Monte Carlo)."""
    rng = random.Random(seed)
    ks = list(ks)
    rankings = []
    trues = []
    for _ in range(n_examples):
        true_idx = rng.randint(0, n_candidates - 1)
        perm = list(range(n_candidates))
        rng.shuffle(perm)
        rankings.append(perm)
        trues.append(true_idx)
    return evaluate_ranked_lists(rankings, trues, ks)

