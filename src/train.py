"""Train KV adapter with layer-level injection (P-Tuning v2 style).

Instead of prepending a soft token at the input embedding layer, this script
trains an adapter whose output is injected as a prefix hidden state at a
specific transformer decoder layer.  The layer's own attention projections
(k_proj, v_proj) create the K/V entries that real tokens attend to.

Training objective is identical to train_adapter.py: teacher-forced
cross-entropy on movie titles through a frozen LLaMA.  The only difference
is WHERE the adapter output enters the model.

Example:
    python scripts/train_kv_adapter.py --smoke
    python scripts/train_kv_adapter.py --target-layer 6 --epochs 10
    python scripts/train_kv_adapter.py --target-layer 8 --n-prefix 2 --epochs 10
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

from src.data import IdMaps, pick_device
from src.kv_adapter import KVAdapter, KVAdapterConfig, install_layer_injection


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--train-csv", type=Path, default=ROOT / "train.csv")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8,
                   help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--sasrec-dim", type=int, default=50)
    p.add_argument("--llm-dim", type=int, default=2048,
                   help="Llama-3.2-1B hidden size (auto-detected)")
    p.add_argument("--hidden-dim", type=int, default=1024,
                   help="Adapter MLP intermediate width")
    p.add_argument("--target-layer", type=int, default=6,
                   help="Decoder layer index for KV injection (0-indexed)")
    p.add_argument("--n-prefix", type=int, default=1,
                   help="Number of virtual prefix tokens to inject")
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--llm-model", type=str,
                   default="unsloth/Llama-3.2-1B-Instruct")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", type=Path, default=None,
                   help="Output path (default: checkpoints/kv_adapter.pt)")
    return p.parse_args()

def _dtype_for_device(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def build_movie_texts(train_csv: Path, id_maps: IdMaps) -> dict[int, str]:
    """idx -> title string (same logic as train_adapter.py)."""
    df = pd.read_csv(
        train_csv, usecols=["MovieID", "Title"],
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

def train_one_movie(
    kv_adapter: KVAdapter,
    llm: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    injected_layer,               # InjectedDecoderLayer wrapper
    e: torch.Tensor,              # (1, sasrec_dim)
    text: str,                    # movie title
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Forward one movie through frozen LLM with layer injection.

    Flow:
        1. KVAdapter projects SASRec embedding → prefix hidden states
        2. InjectedDecoderLayer prepends them at the target layer
        3. LLM teacher-forces the title tokens
        4. CE loss on title prediction; grads flow back through adapter

    The input to the LLM is just the title tokens (with BOS from
    add_special_tokens).  There is NO soft token at the input level.
    The collaborative signal enters only through layer injection.
    """
    # Project the SASRec embedding into the prefix hidden states the layer expects.
    prefix_hs = kv_adapter(e).to(dtype=dtype)       # (1, n_prefix, llm_dim)

    # Tokenize the movie title.
    tokens = tokenizer(text, add_special_tokens=True, return_tensors="pt")
    input_ids = tokens.input_ids.to(device)          # (1, T)

    # Mask out the BOS token from the loss — the adapter prefix carries the
    # "which movie is this" signal, so BOS is just a structural anchor.
    labels = input_ids.clone()
    labels[:, 0] = -100

    # Set the prefix on the injected layer, run the forward pass, then always
    # clear it — even if the forward throws — so we don't leak state.
    injected_layer.set_prefix(prefix_hs)
    try:
        outputs = llm(
            input_ids=input_ids,
            attention_mask=tokens.attention_mask.to(device),
            labels=labels,
            return_dict=True,
        )
    finally:
        injected_layer.clear_prefix()

    return outputs.loss


@torch.no_grad()
def eval_ce(
    kv_adapter: KVAdapter,
    llm: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    injected_layer,
    item_emb: torch.Tensor,
    movie_texts: dict[int, str],
    val_indices: list[int],
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    """Average CE over validation movies."""
    kv_adapter.eval()
    total_loss = 0.0
    n = 0
    for idx in val_indices:
        if idx not in movie_texts:
            continue
        text = movie_texts[idx]
        e = item_emb[idx - 1].unsqueeze(0).to(device, dtype=torch.float32)

        prefix_hs = kv_adapter(e).to(dtype=dtype)
        tokens = tokenizer(text, add_special_tokens=True, return_tensors="pt")
        input_ids = tokens.input_ids.to(device)
        labels = input_ids.clone()
        labels[:, 0] = -100

        injected_layer.set_prefix(prefix_hs)
        try:
            outputs = llm(
                input_ids=input_ids,
                attention_mask=tokens.attention_mask.to(device),
                labels=labels,
                return_dict=True,
            )
        finally:
            injected_layer.clear_prefix()

        total_loss += outputs.loss.item()
        n += 1

    return total_loss / max(n, 1)

def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 3)
        args.batch_size = min(args.batch_size, 4)

    torch.manual_seed(args.seed)
    device = pick_device()
    dtype = _dtype_for_device(device)
    ckpt_dir = args.checkpoint_dir

    # Load the precomputed SASRec item embeddings.
    emb_path = ckpt_dir / "item_embeddings.pt"
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing {emb_path}. Run: python scripts/train_sasrec.py")
    item_emb = torch.load(emb_path, map_location="cpu", weights_only=True).float()
    # If the saved embeddings have a different dim than the default, use the real one.
    if item_emb.shape[1] != args.sasrec_dim:
        args.sasrec_dim = item_emb.shape[1]
        print(f"[data] auto sasrec_dim={args.sasrec_dim}")

    n_items = item_emb.size(0)
    print(f"[device] {device}  dtype={dtype}")
    print(f"[data] {n_items} item embeddings, dim={item_emb.shape[1]}")

    # Load movie metadata so we can look up titles by index.
    id_maps = IdMaps.from_json(ckpt_dir / "id_maps.json")
    movie_texts = build_movie_texts(args.train_csv, id_maps)
    print(f"[data] {len(movie_texts)} movies with title text")

    # Split into train and val, shuffled so the val set is a random sample.
    rng = random.Random(args.seed)
    all_indices = [i for i in range(1, id_maps.n_items + 1) if i in movie_texts]
    rng.shuffle(all_indices)
    n_val = max(1, int(len(all_indices) * args.val_fraction))
    val_indices = all_indices[:n_val]
    train_indices = all_indices[n_val:]

    if args.smoke:
        train_indices = train_indices[:50]
        val_indices = val_indices[:10]

    print(f"[split] {len(train_indices)} train / {len(val_indices)} val movies")

    # Load the LLM and freeze all its parameters — only the adapter trains.
    print(f"[llm] loading {args.llm_model} ...")
    from src.inject_llm import load_llm_and_tokenizer

    llm, tokenizer = load_llm_and_tokenizer(args.llm_model, device, dtype)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    llm.eval()
    for p in llm.parameters():
        p.requires_grad = False

    # Detect the actual hidden size in case the model differs from the default arg.
    base = getattr(llm, "base_model", None)
    llm_hidden = (base.config.hidden_size if base and hasattr(base, "config")
                  else llm.config.hidden_size)
    if llm_hidden != args.llm_dim:
        print(f"[llm] overriding llm_dim: {args.llm_dim} → {llm_hidden}")
        args.llm_dim = llm_hidden

    n_layers = llm.config.num_hidden_layers
    if args.target_layer >= n_layers:
        raise ValueError(
            f"--target-layer {args.target_layer} >= num_layers {n_layers}"
        )
    print(f"[llm] ready  hidden={llm_hidden}  layers={n_layers}")

    # Wrap the target decoder layer so it can accept prefix hidden states.
    injected_layer = install_layer_injection(
        llm, target_layer=args.target_layer, n_prefix=args.n_prefix,
    )

    # Build the KV adapter and set up the optimizer — this is the only thing that trains.
    config = KVAdapterConfig(
        sasrec_dim=args.sasrec_dim,
        llm_dim=args.llm_dim,
        hidden_dim=args.hidden_dim,
        n_prefix=args.n_prefix,
        target_layer=args.target_layer,
    )
    kv_adapter = KVAdapter(config).to(device)
    optimizer = torch.optim.Adam(kv_adapter.parameters(), lr=args.lr)

    n_params = sum(p.numel() for p in kv_adapter.parameters() if p.requires_grad)
    print(
        f"[kv_adapter] {args.sasrec_dim} → {args.hidden_dim} → "
        f"{args.n_prefix}×{args.llm_dim}  "
        f"@ layer {args.target_layer}  ({n_params:,} params)"
    )
    print(f"[train] epochs={args.epochs}  batch_size={args.batch_size}  lr={args.lr}")

    # Training loop. We use gradient accumulation since each forward pass is over
    # a single movie, so batch_size controls how many we accumulate before stepping.
    best_val = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        kv_adapter.train()
        rng.shuffle(train_indices)

        running_loss = 0.0
        n_steps = 0
        t0 = time.time()
        optimizer.zero_grad()
        accum_loss = 0.0
        accum_count = 0

        for i, idx in enumerate(tqdm(train_indices, desc=f"epoch {epoch}", leave=False)):
            text = movie_texts[idx]
            e = item_emb[idx - 1].unsqueeze(0).to(device, dtype=torch.float32)

            loss = train_one_movie(
                kv_adapter, llm, tokenizer, injected_layer,
                e, text, device, dtype,
            )
            (loss / args.batch_size).backward()
            accum_loss += loss.item()
            accum_count += 1

            # Step once we've accumulated enough gradients, or hit the end of the epoch.
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
            kv_adapter, llm, tokenizer, injected_layer,
            item_emb, movie_texts, val_indices, device, dtype,
        )
        elapsed = time.time() - t0

        # Keep a copy of the best weights so we can restore them at the end.
        improved = ""
        if val_ce < best_val:
            best_val = val_ce
            best_state = {k: v.cpu().clone() for k, v in kv_adapter.state_dict().items()}
            improved = " *"

        print(
            f"epoch {epoch}/{args.epochs}  "
            f"train_ce={train_ce:.4f}  val_ce={val_ce:.4f}  "
            f"best_val={best_val:.4f}  ({elapsed:.1f}s){improved}"
        )

    # Restore the best checkpoint before saving so we don't write the final epoch's weights.
    if best_state is not None:
        kv_adapter.load_state_dict(best_state)

    out_path = args.output or (ckpt_dir / "kv_adapter.pt")
    torch.save(
        {
            "model_state_dict": kv_adapter.state_dict(),
            "config": {
                "sasrec_dim": args.sasrec_dim,
                "llm_dim": args.llm_dim,
                "hidden_dim": args.hidden_dim,
                "n_prefix": args.n_prefix,
                "target_layer": args.target_layer,
                "best_val_ce": best_val,
                "training_method": "kv_layer_injection_ce",
                "llm_model": args.llm_model,
            },
        },
        out_path,
    )
    print(f"\n[done] saved {out_path}  (best val_ce={best_val:.4f})")
    print(f"[done] adapter trained with layer-{args.target_layer} KV injection")
    print(f"[next] try different layers:  --target-layer 4 / 8 / 14")


if __name__ == "__main__":
    main()