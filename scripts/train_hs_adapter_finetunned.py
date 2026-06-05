#!/usr/bin/env python3
"""Train hidden-state injection adapter for ranking (layer-targeted).

Instead of injecting soft tokens at the input embedding layer (Mode C), this
adapter learns to produce vectors that are *added* to candidate hidden states
at a specific transformer layer via a forward hook:

    h_new[candidate_pos] = h[candidate_pos] + alpha * adapter(e_movie)

The loss is the same listwise ranking CE over A-J letter logprobs as
train_adapter_ranking.py. Gradients flow through the hook back to the adapter.

Files required in your project directory:
    checkpoints/adapter.pt            (warm-start weights)
    checkpoints/item_embeddings.pt    (SASRec embeddings)
    checkpoints/id_maps.json          (movie ↔ index mappings)
    train.csv                         (training interactions)
    test_ranking.csv                  (eval data)
    test_ranking_prompts.json         (eval prompts)
    src/                              (project source modules)

Google Colab usage (A100 ~15-30 min):
    # Mount Drive, cd to project dir, then:
    !pip install -q -r requirements.txt
    !python scripts/train_hs_adapter.py --epochs 5 --target-layer 8

    # Smoke test:
    !python scripts/train_hs_adapter.py --smoke --target-layer 8

    # Evaluate the trained adapter:
    !python scripts/eval_hidden_state.py \\
        --adapter checkpoints/adapter_hs_L8.pt \\
        --target-layer 8 --alpha 1.0
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adapter import AdapterConfig, EmbeddingAdapter
from src.data import IdMaps, load_interactions, pick_device
from src.inject_llm import (
    LETTERS,
    load_llm_and_tokenizer,
    render_mode_a_prompt,
)
from src.metrics import evaluate_ranked_lists
from src.ranking_data import RankingExample, build_train_ranking_examples, load_ranking_examples


# ══════════════════════════════════════════════════════════════════
#  Autograd-safe hidden-state injection hook
# ══════════════════════════════════════════════════════════════════

class TrainableHSInjector:
    """Forward hook that adds adapter vectors to candidate hidden states.

    Unlike the eval-only HiddenStateInjector, this version uses only
    out-of-place operations so that gradients flow cleanly back through
    the hook into the adapter parameters.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._positions: List[int] = []
        self._vectors: List[torch.Tensor] = []
        self._handle = None

    def install(self, llm: nn.Module, layer_idx: int):
        # Try different model structures until layers are found
        for attr_path in [
            lambda: llm.model.layers,                    # base LlamaForCausalLM
            lambda: llm.model.model.layers,              # PeftModel → LlamaForCausalLM
            lambda: llm.base_model.model.model.layers,   # PeftModel alt path
        ]:
            try:
                layers = attr_path()
                self._handle = layers[layer_idx].register_forward_hook(self._hook_fn)
                return
            except AttributeError:
                continue
        raise AttributeError("Cannot locate decoder layers in model")
    
    def _hook_fn(self, module, input, output):
        if not self._positions:
            return output

        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # ── Build perturbation with out-of-place ops for autograd ──
        if hidden_states.dim() == 3:
            B, T, D = hidden_states.shape
        elif hidden_states.dim() == 2:
            T, D = hidden_states.shape
            B = None
        else:
            return output

        device = hidden_states.device
        dtype = hidden_states.dtype

        # Stack all candidate vectors: (N, D), keeps gradient graph
        casted = [
            self.alpha * z.to(device=device, dtype=dtype)
            for z in self._vectors
        ]
        stacked = torch.stack(casted)                       # (N, D)
        pos_t = torch.tensor(self._positions, dtype=torch.long, device=device)

        # Scatter into a (T, D) delta — differentiable w.r.t. stacked
        delta_flat = torch.zeros(T, D, device=device, dtype=dtype)
        idx = pos_t.unsqueeze(1).expand_as(stacked)         # (N, D)
        delta_flat = delta_flat.scatter_add(0, idx, stacked) # out-of-place

        if B is not None:
            delta = delta_flat.unsqueeze(0).expand(B, -1, -1)
        else:
            delta = delta_flat

        hidden_states = hidden_states + delta                # out-of-place

        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return hidden_states

    def set_candidates(self, positions: List[int], vectors: List[torch.Tensor]):
        self._positions = positions
        self._vectors = vectors

    def clear(self):
        self._positions = []
        self._vectors = []

    def remove(self):
        if self._handle:
            self._handle.remove()
            self._handle = None


# ══════════════════════════════════════════════════════════════════
#  Utilities (same as eval_hidden_state.py)
# ══════════════════════════════════════════════════════════════════

def find_candidate_positions(
    prompt: str, tokenizer, n_candidates: int,
) -> List[int]:
    """Find the token position of each candidate letter label in the prompt."""
    full_ids = tokenizer(
        prompt, add_special_tokens=True, return_tensors="pt"
    ).input_ids[0]

    positions = []
    for i in range(n_candidates):
        letter = LETTERS[i]
        marker = f"\n{letter}."
        char_idx = prompt.find(marker)
        if char_idx == -1:
            marker = f"{letter}."
            char_idx = prompt.find(marker)
        if char_idx == -1:
            positions.append(-1)
            continue

        prefix_text = prompt[: char_idx + 1]
        prefix_ids = tokenizer(
            prefix_text, add_special_tokens=True, return_tensors="pt"
        ).input_ids[0]
        pos = len(prefix_ids)
        if pos >= len(full_ids):
            pos = len(full_ids) - 1
        positions.append(pos)

    return positions


def get_letter_token_ids(tokenizer, n: int) -> torch.Tensor:
    ids = []
    for ch in LETTERS[:n]:
        toks = tokenizer.encode(f" {ch}", add_special_tokens=False)
        if len(toks) != 1:
            raise RuntimeError(f"Letter ' {ch}' tokenises to {toks}")
        ids.append(toks[0])
    return torch.tensor(ids, dtype=torch.long)


# ══════════════════════════════════════════════════════════════════
#  Training forward pass
# ══════════════════════════════════════════════════════════════════

def train_step(
    llm: nn.Module,
    tokenizer,
    adapter: EmbeddingAdapter,
    injector: TrainableHSInjector,
    example: RankingExample,
    item_embeddings: torch.Tensor,
    id_maps: IdMaps,
    letter_ids: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """One forward pass → ranking CE loss (scalar, in computation graph)."""

    n = len(example.candidate_movie_ids)

    # ── Build text prompt ─────────────────────────────────────────
    prompt = render_mode_a_prompt(example)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # ── Find candidate positions ──────────────────────────────────
    positions = find_candidate_positions(prompt, tokenizer, n)

    # ── Project candidate embeddings (WITH gradient) ──────────────
    vectors = []
    for mid in example.candidate_movie_ids:
        idx = id_maps.movie_to_idx.get(int(mid))
        if idx is None:
            vectors.append(torch.zeros(adapter.config.llm_dim, device=device))
        else:
            e = item_embeddings[idx - 1].unsqueeze(0).to(device, dtype=torch.float32)
            z = adapter.project(e)          # (1, llm_dim) — in graph
            vectors.append(z.squeeze(0))    # (llm_dim,)

    # ── Filter invalid positions ──────────────────────────────────
    valid_pos, valid_vec = [], []
    for pos, vec in zip(positions, vectors):
        if pos >= 0:
            valid_pos.append(pos)
            valid_vec.append(vec)

    # ── Inject and run forward ────────────────────────────────────
    injector.set_candidates(valid_pos, valid_vec)
    outputs = llm(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        return_dict=True,
    )
    injector.clear()

    # ── Ranking CE loss over letter logprobs ──────────────────────
    logits = outputs.logits[0, -1, :]           # (vocab_size,)
    letter_logits = logits[letter_ids[:n]]      # (n,)
    target_idx = example.true_position - 1      # 0-based

    loss = F.cross_entropy(
        letter_logits.unsqueeze(0).float(),
        torch.tensor([target_idx], dtype=torch.long, device=device),
    )
    return loss


# ══════════════════════════════════════════════════════════════════
#  Validation (same scoring as eval_hidden_state.py)
# ══════════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_hr1(
    llm, tokenizer, adapter, injector,
    examples, item_embeddings, id_maps,
    letter_ids, device, dtype,
) -> Tuple[float, float]:
    """Return (HR@1, MRR) on validation examples."""
    adapter.eval()
    ranked_lists, true_indices = [], []
    n_cand = len(examples[0].candidate_movie_ids)

    for ex in examples:
        n = len(ex.candidate_movie_ids)
        prompt = render_mode_a_prompt(ex)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        positions = find_candidate_positions(prompt, tokenizer, n)

        vectors = []
        for mid in ex.candidate_movie_ids:
            idx = id_maps.movie_to_idx.get(int(mid))
            if idx is None:
                vectors.append(torch.zeros(adapter.config.llm_dim, device=device))
            else:
                e = item_embeddings[idx - 1].unsqueeze(0).to(device, dtype=torch.float32)
                z = adapter.project(e).detach()
                vectors.append(z.squeeze(0))

        valid_pos, valid_vec = [], []
        for pos, vec in zip(positions, vectors):
            if pos >= 0:
                valid_pos.append(pos)
                valid_vec.append(vec)

        injector.set_candidates(valid_pos, valid_vec)
        outputs = llm(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            return_dict=True,
        )
        injector.clear()

        logits = outputs.logits[0, -1, :]
        lids = letter_ids[:n].to(device)
        probs = torch.softmax(logits[lids].float(), dim=0)
        ranked = torch.argsort(probs, descending=True).tolist()

        ranked_lists.append(ranked)
        true_indices.append(ex.true_position - 1)

    m = evaluate_ranked_lists(ranked_lists, true_indices, ks=[1, 3, 5])
    adapter.train()
    return m["HR@1"], m.get("MRR", 0.0)


# ══════════════════════════════════════════════════════════════════
#  CLI & main
# ══════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--train-csv", type=Path, default=ROOT / "train.csv")
    p.add_argument(
        "--model", type=str, default="unsloth/Llama-3.2-1B-Instruct",
        help="Frozen LLM: HF model id or local PEFT folder",
    )
    p.add_argument(
        "--init-from", type=Path, default=None,
        help="Warm-start adapter weights (default: checkpoints/adapter.pt)",
    )
    p.add_argument("--target-layer", type=int, default=8)
    p.add_argument(
        "--alpha", type=float, default=1.0,
        help="Injection scale during training. Adapter learns to match this.",
    )
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=4, help="Gradient accumulation steps")
    p.add_argument("--max-train", type=int, default=4000)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--n-history", type=int, default=10)
    p.add_argument("--n-candidates", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def _dtype_for_device(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def main():
    args = parse_args()
    if args.smoke:
        args.epochs = 1
        args.max_train = 20

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device()
    dtype = _dtype_for_device(device)
    ckpt_dir = args.checkpoint_dir

    print(f"[device] {device}  dtype={dtype}")
    print(f"[config] layer={args.target_layer}  alpha={args.alpha}  lr={args.lr}")
    print(f"[config] epochs={args.epochs}  batch_size={args.batch_size}  max_train={args.max_train}")

    # ── Load data ─────────────────────────────────────────────────
    id_maps = IdMaps.from_json(ckpt_dir / "id_maps.json")
    item_emb = torch.load(
        ckpt_dir / "item_embeddings.pt", map_location="cpu", weights_only=True
    ).float()
    print(f"[data] {item_emb.size(0)} item embeddings, dim={item_emb.shape[1]}")

    # ── Build training ranking examples ───────────────────────────
    train_df = load_interactions(args.train_csv, min_rating=0.0)
    movie_meta = (
        pd.read_csv(
            args.train_csv,
            usecols=["MovieID", "Title", "Genres"],
            encoding="utf-8",
            encoding_errors="replace",
        ).drop_duplicates("MovieID")
    )
    all_examples = build_train_ranking_examples(
        train_df, id_maps, movie_meta,
        n_history=args.n_history,
        n_candidates=args.n_candidates,
        max_examples=args.max_train,
        seed=args.seed,
    )
    rng = random.Random(args.seed)
    rng.shuffle(all_examples)

    n_val = max(1, int(len(all_examples) * args.val_fraction))
    val_examples = all_examples[:n_val]
    train_examples = all_examples[n_val:]
    print(f"[data] {len(train_examples)} train  /  {len(val_examples)} val ranking examples")

    # ── Load frozen LLM ───────────────────────────────────────────
    print(f"[llm] loading {args.model}...")
    llm, tokenizer = load_llm_and_tokenizer(args.model, device, dtype)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm.eval()
    for p in llm.parameters():
        p.requires_grad = False

    n_layers = llm.config.num_hidden_layers
    llm_hidden = llm.config.hidden_size
    if args.target_layer >= n_layers:
        raise ValueError(f"--target-layer {args.target_layer} >= {n_layers}")
    print(f"[llm] ready  layers={n_layers}  hidden={llm_hidden}")

    # ── Build adapter (warm-start from existing checkpoint) ───────
    init_path = args.init_from or (ckpt_dir / "adapter.pt")
    if init_path.exists():
        ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        adapter = EmbeddingAdapter(
            AdapterConfig(
                sasrec_dim=cfg.get("sasrec_dim", item_emb.shape[1]),
                llm_dim=cfg.get("llm_dim", llm_hidden),
                hidden_dim=cfg.get("hidden_dim", 1024),
            )
        )
        state = ckpt.get("model_state_dict") or ckpt.get("adapter_state_dict")
        if state:
            adapter.load_state_dict(state)
        print(f"[adapter] warm-start from {init_path.name}")
    else:
        adapter = EmbeddingAdapter(
            AdapterConfig(
                sasrec_dim=item_emb.shape[1],
                llm_dim=llm_hidden,
                hidden_dim=1024,
            )
        )
        print("[adapter] random init (no warm-start checkpoint found)")

    adapter.to(device).train()
    n_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"[adapter] {n_params:,} trainable params")

    # ── Pre-compute letter token IDs ──────────────────────────────
    letter_ids = get_letter_token_ids(tokenizer, args.n_candidates).to(device)

    # ── Install hook ──────────────────────────────────────────────
    injector = TrainableHSInjector(alpha=args.alpha)
    injector.install(llm, layer_idx=args.target_layer)
    print(f"[hook] layer {args.target_layer}, alpha={args.alpha}")

    # ── Optimizer ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(adapter.parameters(), lr=args.lr)

    # ── Training loop ─────────────────────────────────────────────
    best_hr1 = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        adapter.train()
        rng.shuffle(train_examples)

        running_loss = 0.0
        n_steps = 0
        accum_loss = 0.0
        accum_count = 0
        t0 = time.time()
        optimizer.zero_grad()

        pbar = tqdm(train_examples, desc=f"epoch {epoch}/{args.epochs}")
        for i, ex in enumerate(pbar):
            if len(ex.candidate_movie_ids) != args.n_candidates:
                continue

            loss = train_step(
                llm, tokenizer, adapter, injector,
                ex, item_emb, id_maps, letter_ids,
                device, dtype,
            )

            (loss / args.batch_size).backward()
            accum_loss += loss.item()
            accum_count += 1

            if accum_count >= args.batch_size or (i + 1) == len(train_examples):
                optimizer.step()
                optimizer.zero_grad()
                running_loss += accum_loss
                n_steps += accum_count
                accum_loss = 0.0
                accum_count = 0

            if (i + 1) % 100 == 0:
                pbar.set_postfix(loss=running_loss / max(n_steps, 1))

        avg_loss = running_loss / max(n_steps, 1)
        elapsed = time.time() - t0

        # ── Validate ──────────────────────────────────────────────
        print(f"  epoch {epoch} train_loss={avg_loss:.4f} ({elapsed:.1f}s) — evaluating...",
              flush=True)
        val_hr1, val_mrr = eval_hr1(
            llm, tokenizer, adapter, injector,
            val_examples, item_emb, id_maps, letter_ids,
            device, dtype,
        )

        improved = ""
        if val_hr1 > best_hr1:
            best_hr1 = val_hr1
            best_state = {k: v.cpu().clone() for k, v in adapter.state_dict().items()}
            improved = " *best*"

        print(
            f"epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}  "
            f"val_HR@1={val_hr1:.4f}  val_MRR={val_mrr:.4f}  "
            f"best_HR@1={best_hr1:.4f}{improved}"
        )

    # ── Save ──────────────────────────────────────────────────────
    injector.remove()

    if best_state is not None:
        adapter.load_state_dict(best_state)

    out_path = args.output or (
        Path("finetuned") / f"adapter_hs_L{args.target_layer}.pt"
    )
    torch.save(
        {
            "model_state_dict": adapter.state_dict(),
            "config": {
                "sasrec_dim": adapter.config.sasrec_dim,
                "llm_dim": adapter.config.llm_dim,
                "hidden_dim": adapter.config.hidden_dim,
                "training_method": "hidden_state_ranking",
                "target_layer": args.target_layer,
                "alpha": args.alpha,
                "llm_model": args.model,
                "best_val_hr1": best_hr1,
                "n_candidates": args.n_candidates,
                "n_history": args.n_history,
            },
        },
        out_path,
    )
    print(f"\n[done] saved {out_path}  (best val_HR@1={best_hr1:.4f})")
    print(f"[next] evaluate with:")
    print(f"  python scripts/eval_hidden_state.py \\")
    print(f"    --adapter {out_path} \\")
    print(f"    --target-layer {args.target_layer} --alpha {args.alpha}")


if __name__ == "__main__":
    main()
