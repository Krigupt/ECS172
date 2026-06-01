#!/usr/bin/env python3
"""Phase 2: Train adapter (SASRec dim -> Llama dim) with LLaMA-grounded CE loss.

For each movie, injects adapter(sasrec_embedding) as a soft prefix token into
frozen LLaMA and teacher-forces the movie's "Title (Genres)" text.
Loss = cross-entropy over the title tokens only.

This grounds the adapter in LLaMA's representation space so the injected
vectors are semantically meaningful to LLaMA at inference time.

Example:
    python scripts/train_adapter.py --smoke
    python scripts/train_adapter.py --epochs 10
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adapter import AdapterConfig, EmbeddingAdapter
from src.data import IdMaps, pick_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--train-csv", type=Path, default=ROOT / "train.csv")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8,
                   help="Number of movies per gradient step (micro-batch)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--sasrec-dim", type=int, default=50)
    p.add_argument("--llm-dim", type=int, default=2048, help="Llama-3.2-1B hidden size")
    p.add_argument("--hidden-dim", type=int, default=1024, help="Adapter MLP intermediate size")
    p.add_argument("--val-fraction", type=float, default=0.1,
                   help="Fraction of movies held out for validation")
    p.add_argument("--llm-model", type=str, default="unsloth/Llama-3.2-1B-Instruct",
                   help="HuggingFace model ID for frozen LLaMA")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _dtype_for_device(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def build_movie_texts(
    train_csv: Path,
    id_maps: IdMaps,
) -> dict[int, str]:
    """Build idx -> title string for all movies in id_maps.

    Only the movie title is used as the reconstruction target — no genres.
    Rationale: the adapter's job is to place the SASRec collaborative vector
    in LLaMA's semantic space for that movie. The title is the cleanest,
    most unambiguous signal for that. Genres are noisy (many movies share
    the same genre string) and would dilute the per-movie specificity of
    the reconstruction target.
    """
    df = pd.read_csv(
        train_csv,
        usecols=["MovieID", "Title"],
    ).drop_duplicates("MovieID")
    meta = df.set_index("MovieID")

    texts: dict[int, str] = {}
    for idx in range(1, id_maps.n_items + 1):
        mid = id_maps.idx_to_movie.get(idx)
        if mid is None or mid not in meta.index:
            continue
        row = meta.loc[mid]
        texts[idx] = str(row["Title"]) if "Title" in row else str(row.title)
    return texts


@torch.no_grad()
def eval_ce(
    adapter: EmbeddingAdapter,
    llm: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    embed_layer: torch.nn.Module,
    item_emb: torch.Tensor,
    movie_texts: dict[int, str],
    val_indices: list[int],
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    """Average CE loss over validation movies."""
    adapter.eval()
    total_loss = 0.0
    n = 0
    for idx in val_indices:
        if idx not in movie_texts:
            continue
        text = movie_texts[idx]
        e = item_emb[idx - 1].unsqueeze(0).to(device, dtype=torch.float32)
        z = adapter.project(e).to(dtype=dtype)  # (1, llm_dim)

        token_ids = tokenizer(
            text, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(device)  # (1, T)
        token_embs = embed_layer(token_ids).to(dtype=dtype)  # (1, T, llm_dim)

        # [soft_token | title_tokens]
        inputs_embeds = torch.cat([z.unsqueeze(1), token_embs], dim=1)  # (1, 1+T, D)

        # Labels: -100 for soft token, then title token IDs
        ignore = torch.full((1, 1), -100, dtype=torch.long, device=device)
        labels = torch.cat([ignore, token_ids], dim=1)  # (1, 1+T)

        outputs = llm(inputs_embeds=inputs_embeds, labels=labels, return_dict=True)
        total_loss += outputs.loss.item()
        n += 1

    return total_loss / max(n, 1)


def train_one_movie(
    adapter: EmbeddingAdapter,
    llm: AutoModelForCausalLM,
    embed_layer: torch.nn.Module,
    tokenizer: AutoTokenizer,
    e: torch.Tensor,
    text: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Forward pass for one movie; returns scalar loss (in graph)."""
    z = adapter.project(e).to(dtype=dtype)  # (1, llm_dim)

    token_ids = tokenizer(
        text, add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(device)  # (1, T)

    # Get LLaMA token embeddings (no grad needed for the text part)
    with torch.no_grad():
        token_embs = embed_layer(token_ids).to(dtype=dtype)  # (1, T, D)

    # [soft_token | title_tokens]
    inputs_embeds = torch.cat([z.unsqueeze(1), token_embs], dim=1)  # (1, 1+T, D)

    # Labels: -100 for soft token position, then actual token IDs
    ignore = torch.full((1, 1), -100, dtype=torch.long, device=device)
    labels = torch.cat([ignore, token_ids], dim=1)  # (1, 1+T)

    outputs = llm(inputs_embeds=inputs_embeds, labels=labels, return_dict=True)
    return outputs.loss


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 3)
        args.batch_size = min(args.batch_size, 4)

    torch.manual_seed(args.seed)
    device = pick_device()
    dtype = _dtype_for_device(device)
    ckpt_dir = args.checkpoint_dir

    # ------------------------------------------------------------------
    # Load SASRec item embeddings
    # ------------------------------------------------------------------
    emb_path = ckpt_dir / "item_embeddings.pt"
    if not emb_path.exists():
        raise FileNotFoundError(
            f"Missing {emb_path}. Run: python scripts/train_sasrec.py"
        )
    item_emb = torch.load(emb_path, map_location="cpu", weights_only=True).float()
    if item_emb.shape[1] != args.sasrec_dim:
        args.sasrec_dim = item_emb.shape[1]
        print(f"[data] using sasrec_dim={args.sasrec_dim} from checkpoint")

    n_items = item_emb.size(0)
    print(f"[device] {device} (dtype={dtype})")
    print(f"[data] {n_items} item embeddings, dim={item_emb.shape[1]}")

    # ------------------------------------------------------------------
    # Load movie metadata (Title + Genres)
    # ------------------------------------------------------------------
    id_maps = IdMaps.from_json(ckpt_dir / "id_maps.json")
    movie_texts = build_movie_texts(args.train_csv, id_maps)
    print(f"[data] {len(movie_texts)} movies with title+genre text")

    # ------------------------------------------------------------------
    # Train / val split (item-level)
    # ------------------------------------------------------------------
    rng = random.Random(args.seed)
    all_indices = [idx for idx in range(1, id_maps.n_items + 1) if idx in movie_texts]
    rng.shuffle(all_indices)
    n_val = max(1, int(len(all_indices) * args.val_fraction))
    val_indices = all_indices[:n_val]
    train_indices = all_indices[n_val:]

    if args.smoke:
        train_indices = train_indices[:50]
        val_indices = val_indices[:10]

    print(f"[split] {len(train_indices)} train movies, {len(val_indices)} val movies")

    # ------------------------------------------------------------------
    # Load frozen LLaMA (Supports base HF models or local PEFT LoRA)
    # ------------------------------------------------------------------
    print(f"[llm] loading {args.llm_model} (first run downloads ~2-5 min)...")
    from src.inject_llm import load_llm_and_tokenizer
    
    llm, tokenizer = load_llm_and_tokenizer(args.llm_model, device, dtype)
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm.eval()
    for p in llm.parameters():
        p.requires_grad = False

    # Resolve actual hidden size from the loaded model
    llm_hidden = llm.config.hidden_size
    if llm_hidden != args.llm_dim:
        print(f"[llm] overriding llm_dim: {args.llm_dim} -> {llm_hidden}")
        args.llm_dim = llm_hidden

    embed_layer = llm.get_input_embeddings()
    print(f"[llm] ready (hidden_size={llm_hidden})")

    # ------------------------------------------------------------------
    # Build adapter
    # ------------------------------------------------------------------
    config = AdapterConfig(
        sasrec_dim=args.sasrec_dim,
        llm_dim=args.llm_dim,
        hidden_dim=args.hidden_dim,
    )
    adapter = EmbeddingAdapter(config).to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(
        f"[adapter] {args.sasrec_dim} -> {args.hidden_dim} -> {args.llm_dim} "
        f"({n_params:,} trainable params)"
    )
    print(f"[train] epochs={args.epochs}  batch_size={args.batch_size}  lr={args.lr}")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        rng.shuffle(train_indices)

        running_loss = 0.0
        n_steps = 0
        t0 = time.time()

        # Accumulate gradients over batch_size movies
        optimizer.zero_grad()
        accum_loss = 0.0
        accum_count = 0

        for i, idx in enumerate(tqdm(train_indices, desc=f"epoch {epoch}", leave=False)):
            text = movie_texts[idx]
            e = item_emb[idx - 1].unsqueeze(0).to(device, dtype=torch.float32)

            loss = train_one_movie(
                adapter, llm, embed_layer, tokenizer,
                e, text, device, dtype,
            )
            # Scale loss by batch size for gradient accumulation
            (loss / args.batch_size).backward()
            accum_loss += loss.item()
            accum_count += 1

            if accum_count >= args.batch_size or (i + 1) == len(train_indices):
                optimizer.step()
                optimizer.zero_grad()
                running_loss += accum_loss
                n_steps += accum_count
                accum_loss = 0.0
                accum_count = 0

        train_ce = running_loss / max(n_steps, 1)

        # Validation
        val_ce = eval_ce(
            adapter, llm, tokenizer, embed_layer,
            item_emb, movie_texts, val_indices, device, dtype,
        )
        elapsed = time.time() - t0

        improved = ""
        if val_ce < best_val:
            best_val = val_ce
            best_state = {k: v.cpu().clone() for k, v in adapter.state_dict().items()}
            improved = " *"

        print(
            f"epoch {epoch}/{args.epochs}  "
            f"train_ce={train_ce:.4f}  val_ce={val_ce:.4f}  "
            f"best_val={best_val:.4f}  ({elapsed:.1f}s){improved}"
        )

    # ------------------------------------------------------------------
    # Save best checkpoint
    # ------------------------------------------------------------------
    if best_state is not None:
        adapter.load_state_dict(best_state)

    out_path = ckpt_dir / "adapter.pt"
    torch.save(
        {
            "model_state_dict": adapter.state_dict(),
            "config": {
                "sasrec_dim": args.sasrec_dim,
                "llm_dim": args.llm_dim,
                "hidden_dim": args.hidden_dim,
                "best_val_ce": best_val,
                "training_method": "llama_grounded_ce",
                "llm_model": args.llm_model,
            },
        },
        out_path,
    )
    print(f"[done] saved {out_path}  (best val_ce={best_val:.4f})")
    print(f"[done] adapter trained with LLaMA-grounded CE (teacher-forced title reconstruction)")


if __name__ == "__main__":
    main()
