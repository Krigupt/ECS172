#!/usr/bin/env python3
"""Build the letter-based (A-J) prompt JSON from test_ranking_prompts.json.

Emits test_ranking_prompts_AC.json — a human-readable record of exactly what
the eval feeds the model in each condition:

  * prompt_A         — Mode A text baseline (titles relabelled A.-J., 'Answer:' cue)
  * prompt_C_template — Mode C layout with [SOFT_TOKEN movie_id=...] placeholders
                        (the real soft token replaces each placeholder at eval time)
  * true_letter      — the correct answer letter, derived from true_positive_pos

This script is torch-free and needs no GPU / model — it just reformats text and
verifies the ground-truth alignment that the off-by-one trap hides in.

Example:
    python scripts/build_AC_prompts.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LETTERS = "ABCDEFGHIJ"
INJECT_SPLIT = "\n\nFrom the list below"

FRAMING = (
    "\n\nEach candidate below is represented as a collaborative filtering "
    "embedding that encodes viewing patterns from similar users:"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--json", type=Path, default=ROOT / "test_ranking_prompts.json")
    p.add_argument("--output", type=Path, default=ROOT / "test_ranking_prompts_AC.json")
    p.add_argument("--max-samples", type=int, default=None)
    return p.parse_args()


def split_prompt(prompt: str) -> tuple[str, str]:
    if INJECT_SPLIT not in prompt:
        raise ValueError("prompt missing injection split marker")
    prefix, rest = prompt.split(INJECT_SPLIT, 1)
    return prefix, INJECT_SPLIT + rest


def candidate_titles(suffix_text: str) -> list[str]:
    titles = []
    for line in suffix_text.split("\n"):
        m = re.match(r"^\d+\.\s+(.+)$", line.strip())
        if m:
            titles.append(m.group(1))
    return titles


def build_mode_a(prefix: str, titles: list[str]) -> str:
    n = len(titles)
    lines = [f"{LETTERS[i]}. {t}" for i, t in enumerate(titles)]
    return (
        prefix
        + "\n\nFrom the list below, rank which movie this user would most likely enjoy:\n"
        + "\n".join(lines)
        + f"\n\nReply with just the letter (A-{LETTERS[n-1]}) of the movie they would rate highest."
        + "\nAnswer:"
    )


def build_mode_c_template(prefix: str, candidate_ids: list[int]) -> str:
    n = len(candidate_ids)
    lines = [f"{LETTERS[i]}. [SOFT_TOKEN movie_id={mid}]" for i, mid in enumerate(candidate_ids)]
    return (
        prefix
        + FRAMING
        + "\n"
        + "\n".join(lines)
        + f"\n\nReply with just the letter (A-{LETTERS[n-1]}) of the movie they would rate highest."
        + "\nAnswer:"
    )


def main() -> None:
    args = parse_args()
    rows = json.loads(args.json.read_text())
    if args.max_samples:
        rows = rows[: args.max_samples]

    out_rows = []
    mismatches = 0
    for row in rows:
        prefix, suffix = split_prompt(row["prompt"])
        candidates = [int(c) for c in row["candidates"]]
        titles = candidate_titles(suffix)
        n = len(candidates)

        if len(titles) != n:  # fall back to IDs if titles unparseable
            titles = [f"Movie {mid}" for mid in candidates]

        pos = int(row["true_positive_pos"])  # 0-based index into candidates
        true_id = int(row["true_positive_id"])

        # ── Off-by-one guard: the answer letter MUST come from the 0-based
        #    true_positive_pos, and candidates[pos] must equal true_positive_id.
        if not (0 <= pos < n) or candidates[pos] != true_id:
            mismatches += 1
            # recover the real index by searching for the id
            pos = candidates.index(true_id) if true_id in candidates else pos
        true_letter = LETTERS[pos]

        out_rows.append(
            {
                "UserID": int(row["UserID"]),
                "candidate_movie_ids": candidates,
                "candidate_titles": titles,
                "true_letter": true_letter,
                "true_positive_id": true_id,
                "true_positive_Title": row.get("true_positive_Title", ""),
                "prompt_A": build_mode_a(prefix, titles),
                "prompt_C_template": build_mode_c_template(prefix, candidates),
            }
        )

    args.output.write_text(json.dumps(out_rows, indent=2))
    print(f"[done] wrote {len(out_rows)} rows -> {args.output}")
    if mismatches:
        print(f"[warn] {mismatches} rows had true_positive_pos != index(true_id); recovered by search")
    else:
        print("[ok] all rows: candidates[true_positive_pos] == true_positive_id (no off-by-one)")
    # Show one example for eyeballing
    if out_rows:
        ex = out_rows[0]
        print("\n--- sample prompt_A (user", ex["UserID"], ") ---")
        print(ex["prompt_A"])
        print("\n--- sample prompt_C_template ---")
        print(ex["prompt_C_template"])
        print(f"\ntrue_letter = {ex['true_letter']}  ({ex['true_positive_Title']})")


if __name__ == "__main__":
    main()
