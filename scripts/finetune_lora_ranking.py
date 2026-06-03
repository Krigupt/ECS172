#!/usr/bin/env python3
"""Mode A LoRA finetune — the fair text-side counterpart to the ranking adapter.

Trains a LoRA on `unsloth/Llama-3.2-1B-Instruct` for the *exact* task the eval
measures: maximise the probability of the correct letter token (` A`..` J`) at
the `Answer:` position of a Mode A ranking prompt.

Why this is the fair comparison to Mode C's `adapter_ranking.pt`:
  - Same data:   ranking examples come from `build_train_ranking_examples` with
                 the same seed, so the LoRA and the ranking adapter rank the
                 identical candidate sets for the identical train users.
  - Same prompt: prompts are built with `render_mode_a_prompt`, the same function
                 `eval_ranking.py` uses for Mode A — byte-identical, no drift.
  - Same loss:   cross-entropy over the 10 letter logits at the answer position,
                 which is precisely how eval scores candidates.
  - Same eval:   validate (and finally report via `eval_ranking.py --modes A`)
                 on the held-out test users in `test_ranking_prompts.json`.
  - Best ckpt:   selected by val HR@1, mirroring `train_adapter_ranking.py`.

The only variable left versus Mode C is candidate *modality* (text titles vs
soft tokens) and *where* the learning lives (LoRA on the LLM vs the projector).

Example (Colab):
    python scripts/finetune_lora_ranking.py \
        --model unsloth/Llama-3.2-1B-Instruct \
        --epochs 3 --max-train 5000 --val-fraction 0.1 \
        --output ./llama31-1b-movielens-ranking-lora
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import IdMaps, load_interactions, pick_device
from src.inject_llm import LETTERS, render_mode_a_prompt
from src.metrics import evaluate_ranked_lists
from src.ranking_data import build_train_ranking_examples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--train-csv", type=Path, default=ROOT / "train.csv")
    p.add_argument(
        "--model",
        type=str,
        default="unsloth/Llama-3.2-1B-Instruct",
        help="Base LLM (HF id). MUST match the model used at eval / for Mode C.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "llama31-1b-movielens-ranking-lora",
        help="Output PEFT folder. Eval with: eval_ranking.py --modes A --model <this>",
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=8, help="Gradient accumulation steps")
    p.add_argument("--max-train", type=int, default=5000, help="Max training ranking examples")
    p.add_argument("--val-fraction", type=float, default=0.1, help="Fraction held out for val HR@1")
    p.add_argument("--n-history", type=int, default=10)
    p.add_argument("--n-candidates", type=int, default=10, help="Candidates per ranking example")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--use-chat-template",
        action="store_true",
        help="Wrap prompts in the chat template. Default OFF to match eval_ranking.py's "
        "default (raw 'Answer:' cue). If you turn this on, also pass --chat-template to eval.",
    )
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _dtype_for_device(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def resolve_letter_ids(tokenizer, n: int, device: torch.device) -> torch.Tensor:
    """Single token id for each label ' A'..' J'; assert single-token (same as eval)."""
    ids = []
    for ch in LETTERS[:n]:
        toks = tokenizer.encode(f" {ch}", add_special_tokens=False)
        if len(toks) != 1:
            raise RuntimeError(
                f"Letter ' {ch}' tokenises to {toks} (expected single token). "
                "Tokenizer/label scheme mismatch."
            )
        ids.append(toks[0])
    return torch.tensor(ids, dtype=torch.long, device=device)


def make_prompt_text(example, tokenizer, use_chat_template: bool) -> str:
    """Mirror InjectedLlamaRanker.build_mode_a_prompt exactly."""
    raw = render_mode_a_prompt(example)
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": raw}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return raw


def answer_logits(model, tokenizer, example, use_chat_template: bool, device, letter_ids):
    """Forward one prompt; return the logits over the n letter tokens at the answer slot."""
    text = make_prompt_text(example, tokenizer, use_chat_template)
    enc = tokenizer(text, return_tensors="pt").to(device)
    out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask, return_dict=True)
    logits = out.logits[0, -1, :]  # next-token distribution at 'Answer:'
    return logits[letter_ids]  # (n,)


@torch.no_grad()
def eval_hr1(model, tokenizer, examples, use_chat_template, device, letter_ids) -> float:
    model.eval()
    ranked_lists, true_indices = [], []
    for ex in examples:
        ll = answer_logits(model, tokenizer, ex, use_chat_template, device, letter_ids)
        ranked = torch.argsort(ll.float(), descending=True).tolist()
        ranked_lists.append(ranked)
        true_indices.append(ex.true_position - 1)
    model.train()
    return evaluate_ranked_lists(ranked_lists, true_indices, ks=[1])["HR@1"]


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
    print(f"[config] model={args.model}  chat_template={args.use_chat_template}")

    # ── Data: identical ranking examples to the ranking adapter (same seed) ──
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
    print(f"[data] {len(train_examples)} train / {len(val_examples)} val ranking examples")

    # ── Model + LoRA ────────────────────────────────────────────────────
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
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

    letter_ids = resolve_letter_ids(tokenizer, args.n_candidates, device)
    print(f"[letters] token ids for ' A'..' {LETTERS[args.n_candidates-1]}': {letter_ids.tolist()}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    print(f"[train] epochs={args.epochs} accum={args.batch_size} lr={args.lr}")

    best_hr1 = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(train_examples)
        running, n_steps = 0.0, 0
        accum_count = 0
        t0 = time.time()
        optimizer.zero_grad()

        for i, ex in enumerate(tqdm(train_examples, desc=f"epoch {epoch}")):
            if len(ex.candidate_movie_ids) != args.n_candidates:
                continue
            ll = answer_logits(model, tokenizer, ex, args.use_chat_template, device, letter_ids)
            true_idx = torch.tensor([ex.true_position - 1], device=device)
            loss = loss_fn(ll.float().unsqueeze(0), true_idx)

            (loss / args.batch_size).backward()
            running += loss.item()
            n_steps += 1
            accum_count += 1
            if accum_count >= args.batch_size or (i + 1) == len(train_examples):
                optimizer.step()
                optimizer.zero_grad()
                accum_count = 0

        avg = running / max(n_steps, 1)
        print(f"  epoch {epoch} train_loss={avg:.4f} ({time.time()-t0:.1f}s) — val HR@1...", flush=True)
        val_hr1 = eval_hr1(model, tokenizer, val_examples, args.use_chat_template, device, letter_ids)

        flag = ""
        if val_hr1 > best_hr1:
            best_hr1 = val_hr1
            args.output.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(args.output))
            tokenizer.save_pretrained(str(args.output))
            flag = " *best — saved*"
        print(f"epoch {epoch}/{args.epochs} loss={avg:.4f} val_HR@1={val_hr1:.4f} "
              f"best={best_hr1:.4f}{flag}", flush=True)

    print(f"[done] best val_HR@1={best_hr1:.4f}  ->  {args.output}")
    print("[next] python scripts/eval_ranking.py --modes A "
          f"--model {args.output}"
          + (" --chat-template" if args.use_chat_template else ""))


if __name__ == "__main__":
    main()
