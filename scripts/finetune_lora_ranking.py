#!/usr/bin/env python3
"""Mode A LoRA finetune — the fair text-side counterpart to the ranking adapter.

Trains a LoRA on `unsloth/Llama-3.2-1B-Instruct` for the *exact* task the eval
measures: maximise the probability of the correct letter token (` A`..` J`) at
the `Answer:` position of a Mode A ranking prompt.

Why this is the fair comparison to Mode C's `adapter_ranking.pt`:
  - Same data:   ranking examples from `build_train_ranking_examples`, same seed
                 as the ranking adapter => identical candidate sets for the same
                 train users.
  - Same prompt: `_build_mode_a_prompt` mirrors `InjectedLlamaRanker.build_mode_a_prompt`
                 byte-for-byte (same regex title parser, same letter labels, same
                 footer, same Answer: cue).
  - Same loss:   CE over the 10 letter logits at the Answer: position — exactly
                 how eval_ranking.py scores candidates.
  - Same eval:   validate per-epoch by HR@1, save best ckpt, then final report via
                 `eval_ranking.py --modes A --model <output>`.

Example (Colab):
    python scripts/finetune_lora_ranking.py \\
        --model unsloth/Llama-3.2-1B-Instruct \\
        --epochs 3 --max-train 5000 --val-fraction 0.1 \\
        --output ./llama31-1b-movielens-ranking-lora
"""

from __future__ import annotations

import argparse
import random
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import IdMaps, load_interactions, pick_device
from src.metrics import evaluate_ranked_lists
from src.ranking_data import build_train_ranking_examples, RankingExample

LETTERS = "ABCDEFGHIJ"


# ── Prompt builder (mirrors InjectedLlamaRanker.build_mode_a_prompt) ────────

def _parse_titles_from_suffix(suffix_text: str) -> list[str]:
    """Extract 'Title (genres)' from the numbered candidate lines in suffix_text."""
    titles = []
    for line in suffix_text.split("\n"):
        line = line.strip()
        m = re.match(r"^\d+\.\s+(.+)$", line)
        if m:
            titles.append(m.group(1))
    return titles


def _build_mode_a_prompt(example: RankingExample) -> str:
    """Raw Mode A prompt — letters A-J, ends with 'Answer:'.

    Must produce byte-identical output to InjectedLlamaRanker.build_mode_a_prompt
    (with use_chat_template=False, the eval default).
    """
    titles = _parse_titles_from_suffix(example.suffix_text)
    n = len(titles)
    if n == 0:
        titles = [f"Movie {mid}" for mid in example.candidate_movie_ids]
        n = len(titles)
    letter_range = f"{LETTERS[0]}-{LETTERS[n - 1]}"
    cand_lines = [f"{LETTERS[i]}. {title}" for i, title in enumerate(titles)]
    suffix = (
        "\n\nFrom the list below, rank which movie this user would most likely enjoy:\n"
        + "\n".join(cand_lines)
        + f"\n\nReply with just the letter ({letter_range}) of the movie they would rate highest."
        + "\nAnswer:"
    )
    return example.prefix_text + suffix


# ── Helpers ──────────────────────────────────────────────────────────────────

def _dtype_for_device(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def _resolve_letter_ids(tokenizer, n: int, device: torch.device) -> torch.Tensor:
    """Single token id for ' A'..' J'; hard-fails if any is multi-token."""
    ids = []
    for ch in LETTERS[:n]:
        toks = tokenizer.encode(f" {ch}", add_special_tokens=False)
        if len(toks) != 1:
            raise RuntimeError(
                f"Letter ' {ch}' tokenises to {toks} (expected single token). "
                "Tokenizer / label mismatch."
            )
        ids.append(toks[0])
    return torch.tensor(ids, dtype=torch.long, device=device)


def _answer_logits(model, tokenizer, example, device, letter_ids):
    """One forward pass; return the (n,) logits over letter tokens at Answer:."""
    text = _build_mode_a_prompt(example)
    enc = tokenizer(text, return_tensors="pt").to(device)
    out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                return_dict=True)
    return out.logits[0, -1, :][letter_ids]


@torch.no_grad()
def _eval_hr1(model, tokenizer, examples, device, letter_ids) -> float:
    model.eval()
    ranked_lists, true_indices = [], []
    for ex in examples:
        ll = _answer_logits(model, tokenizer, ex, device, letter_ids)
        ranked_lists.append(torch.argsort(ll.float(), descending=True).tolist())
        true_indices.append(ex.true_position - 1)
    model.train()
    return evaluate_ranked_lists(ranked_lists, true_indices, ks=[1])["HR@1"]


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--train-csv", type=Path, default=ROOT / "train.csv")
    p.add_argument("--model", type=str, default="unsloth/Llama-3.2-1B-Instruct",
                   help="Base LLM. Must match the model used for Mode C at eval time.")
    p.add_argument("--output", type=Path,
                   default=ROOT / "llama31-1b-movielens-ranking-lora",
                   help="PEFT output folder. Eval: eval_ranking.py --modes A --model <this>")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=8,
                   help="Gradient accumulation steps before optimizer.step()")
    p.add_argument("--max-train", type=int, default=5000,
                   help="Max ranking examples drawn from train users")
    p.add_argument("--val-fraction", type=float, default=0.1,
                   help="Fraction of examples held out for per-epoch HR@1 validation")
    p.add_argument("--n-history", type=int, default=10)
    p.add_argument("--n-candidates", type=int, default=10)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.max_train = 20

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()
    dtype = _dtype_for_device(device)
    print(f"[device] {device}  dtype={dtype}")
    print(f"[config] model={args.model}  epochs={args.epochs}  "
          f"max_train={args.max_train}  lr={args.lr}")

    # ── Data (identical examples to the ranking adapter, same seed) ─────────
    train_df = load_interactions(args.train_csv, min_rating=0.0)
    movie_meta = (
        pd.read_csv(args.train_csv, usecols=["MovieID", "Title", "Genres"],
                    encoding="utf-8", encoding_errors="replace")
        .drop_duplicates("MovieID")
    )
    id_maps = IdMaps.from_json(args.checkpoint_dir / "id_maps.json")

    all_examples = build_train_ranking_examples(
        train_df, id_maps, movie_meta,
        n_history=args.n_history, n_candidates=args.n_candidates,
        max_examples=args.max_train, seed=args.seed,
    )
    rng = random.Random(args.seed)
    rng.shuffle(all_examples)
    n_val = max(1, int(len(all_examples) * args.val_fraction))
    val_examples = all_examples[:n_val]
    train_examples = all_examples[n_val:]
    print(f"[data] {len(train_examples)} train  /  {len(val_examples)} val")

    # ── Model + LoRA ─────────────────────────────────────────────────────────
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    letter_ids = _resolve_letter_ids(tokenizer, args.n_candidates, device)
    print(f"[letters] ' A'..'{LETTERS[args.n_candidates-1]}' -> {letter_ids.tolist()}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    # ── Training loop ─────────────────────────────────────────────────────────
    best_hr1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train_examples)
        running, n_steps, accum = 0.0, 0, 0
        t0 = time.time()
        optimizer.zero_grad()

        for i, ex in enumerate(tqdm(train_examples, desc=f"epoch {epoch}")):
            if len(ex.candidate_movie_ids) != args.n_candidates:
                continue

            ll = _answer_logits(model, tokenizer, ex, device, letter_ids)
            true_idx = torch.tensor([ex.true_position - 1], device=device)
            loss = loss_fn(ll.float().unsqueeze(0), true_idx)

            (loss / args.batch_size).backward()
            running += loss.item()
            n_steps += 1
            accum += 1

            if accum >= args.batch_size or (i + 1) == len(train_examples):
                optimizer.step()
                optimizer.zero_grad()
                accum = 0

        avg = running / max(n_steps, 1)
        elapsed = time.time() - t0
        print(f"  epoch {epoch} train_loss={avg:.4f} ({elapsed:.1f}s) — val HR@1...",
              flush=True)

        val_hr1 = _eval_hr1(model, tokenizer, val_examples, device, letter_ids)
        flag = ""
        if val_hr1 > best_hr1:
            best_hr1 = val_hr1
            args.output.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(args.output))
            tokenizer.save_pretrained(str(args.output))
            flag = "  *best — saved*"
        print(f"epoch {epoch}/{args.epochs}  loss={avg:.4f}  "
              f"val_HR@1={val_hr1:.4f}  best={best_hr1:.4f}{flag}", flush=True)

    print(f"\n[done] best val_HR@1={best_hr1:.4f}  saved -> {args.output}")
    print(f"[next] python scripts/eval_ranking.py --modes A --model {args.output}")


if __name__ == "__main__":
    main()
