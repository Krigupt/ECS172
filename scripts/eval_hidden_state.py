"""Evaluate hidden-state addition injection for ranking.

Instead of injecting soft tokens at the input embedding layer, this script
adds the adapter-projected SASRec vector directly to candidate hidden states
at a chosen transformer layer using a forward hook.

    h_new[candidate_pos] = h[candidate_pos] + alpha * z_movie

The prompt is pure text (Mode A style with letter labels A-J). The
collaborative signal enters only through the hidden-state addition at the
target layer. Scoring is by log-prob of letter tokens, same as eval_ranking.py.

Works with both the base LLaMA model and a LoRA-finetuned model.

Examples (run from project root):

    # Base model + adapter, inject at layer 8, alpha=0.1
    python scripts/eval_hidden_state.py --target-layer 8 --alpha 0.1

    # LoRA finetuned model
    python scripts/eval_hidden_state.py --model llama31-1b-movielens-full-final --target-layer 8

    # Quick smoke test
    python scripts/eval_hidden_state.py --smoke --target-layer 8

    # Sweep layers
    for L in 1 6 8 10 14; do
        python scripts/eval_hidden_state.py --target-layer $L --alpha 0.1
    done
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adapter import AdapterConfig, EmbeddingAdapter
from src.data import IdMaps, pick_device
from src.inject_llm import (
    DEFAULT_LLM_PATH,
    LETTERS,
    load_llm_and_tokenizer,
    render_mode_a_prompt,
)
from src.metrics import evaluate_ranked_lists, random_baseline
from src.ranking_data import RankingExample, apply_n_candidates, load_ranking_examples

#  Hidden-state injection hook
class HiddenStateInjector:
    """Forward hook that adds adapter vectors to candidate hidden states.

    Usage::

        injector = HiddenStateInjector(alpha=0.1)
        injector.install(llm, layer_idx=8)

        # before each forward pass:
        injector.set_candidates(positions=[120, 132, 144, ...],
                                vectors=[z_A, z_B, z_C, ...])
        outputs = llm(input_ids=...)
        injector.clear()

        # when done:
        injector.remove()
    """

    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self._positions: List[int] = []
        self._vectors: List[torch.Tensor] = []
        self._handle = None

    def install(self, llm: nn.Module, layer_idx: int):
        # Try a few common ways models expose their decoder layers,
        # since the attribute path differs between base and PEFT-wrapped models.
        for attr_path in [
            lambda: llm.model.layers,
            lambda: llm.model.model.layers,
            lambda: llm.base_model.model.model.layers,
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

        # Newer and older versions of transformers return hidden states differently —
        # sometimes it's a plain tensor, sometimes it's wrapped in a tuple.
        if isinstance(output, tuple):
            hidden_states = output[0]
        elif isinstance(output, torch.Tensor):
            hidden_states = output
        else:
            hidden_states = output[0]

        # Add alpha * z directly into the hidden state at each candidate's token position.
        for pos, z in zip(self._positions, self._vectors):
            if hidden_states.dim() == 3:
                hidden_states[:, pos, :] += self.alpha * z.to(
                    device=hidden_states.device, dtype=hidden_states.dtype
                )
            elif hidden_states.dim() == 2:
                hidden_states[pos, :] += self.alpha * z.to(
                    device=hidden_states.device, dtype=hidden_states.dtype
                )

        # Put the output back in whatever format we found it in.
        if isinstance(output, tuple):
            return (hidden_states,) + output[1:]
        return output

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


#  Candidate position finder
def find_candidate_positions(
    prompt: str,
    tokenizer,
    n_candidates: int,
) -> List[int]:
    """Find the token position of each candidate letter label in the prompt.

    The prompt has lines like:
        A. The Matrix (1999) (Action|Sci-Fi)
        B. Titanic (1997) (Drama|Romance)
        ...

    We find where each "\\nX." starts in the tokenized sequence and return
    the position of the letter token for each candidate.
    """
    full_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt").input_ids[0]

    positions = []
    for i in range(n_candidates):
        letter = LETTERS[i]
        # Look for the line break + letter + period that starts each candidate entry.
        marker = f"\n{letter}."
        char_idx = prompt.find(marker)
        if char_idx == -1:
            # The very first candidate might not have a newline before it, so try without.
            marker = f"{letter}."
            char_idx = prompt.find(marker)

        if char_idx == -1:
            positions.append(-1)  # couldn't find this one, skip it
            continue

        # Tokenize everything up to the newline before the letter to figure out
        # which token index the letter itself lands on.
        prefix_text = prompt[:char_idx + 1]  # include the \n
        prefix_ids = tokenizer(prefix_text, add_special_tokens=True, return_tensors="pt").input_ids[0]
        # The letter token comes right after the prefix ends.
        pos = len(prefix_ids)

        # Make sure we don't go out of bounds.
        if pos >= len(full_ids):
            pos = len(full_ids) - 1

        positions.append(pos)

    return positions


#  Scoring
def get_letter_token_ids(tokenizer, n: int) -> torch.Tensor:
    """Get the single token ID for ' A', ' B', ..., ' J'."""
    ids = []
    for ch in LETTERS[:n]:
        toks = tokenizer.encode(f" {ch}", add_special_tokens=False)
        if len(toks) != 1:
            raise RuntimeError(f"Letter ' {ch}' tokenises to {toks} (expected 1 token)")
        ids.append(toks[0])
    return torch.tensor(ids, dtype=torch.long)


@torch.no_grad()
def score_example(
    llm: nn.Module,
    tokenizer,
    adapter: EmbeddingAdapter,
    injector: HiddenStateInjector,
    example: RankingExample,
    item_embeddings: torch.Tensor,
    id_maps: IdMaps,
    device: torch.device,
    dtype: torch.dtype,
    use_chat_template: bool = False,
) -> Tuple[List[int], List[float]]:
    """Score one ranking example and return (ranked_indices, probs)."""

    n = len(example.candidate_movie_ids)

    # Build the text prompt with letter-labeled candidates (A through J).
    prompt = render_mode_a_prompt(example)
    if use_chat_template and hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    # Tokenize the full prompt.
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Figure out which token positions correspond to each candidate's letter label.
    positions = find_candidate_positions(prompt, tokenizer, n)

    # Project each candidate movie's SASRec embedding into the LLM's hidden dimension.
    vectors = []
    for mid in example.candidate_movie_ids:
        idx = id_maps.movie_to_idx.get(int(mid))
        if idx is None:
            # Movie not in our index — fall back to a zero vector so it has no effect.
            vectors.append(torch.zeros(adapter.config.llm_dim, device=device))
        else:
            e = item_embeddings[idx - 1].unsqueeze(0).to(device, dtype=torch.float32)
            z = adapter.project(e).detach()
            vectors.append(z.squeeze(0))  # shape: (llm_dim,)

    # Drop any candidates whose positions we couldn't locate in the token sequence.
    valid_pos = []
    valid_vec = []
    for pos, vec in zip(positions, vectors):
        if pos >= 0:
            valid_pos.append(pos)
            valid_vec.append(vec)

    # Register the candidate info with the hook, run the forward pass, then clear it.
    injector.set_candidates(valid_pos, valid_vec)
    outputs = llm(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        return_dict=True,
    )
    injector.clear()

    # Read off the logits at the last token position and score each letter option.
    logits = outputs.logits[0, -1, :]  # shape: (vocab_size,)
    letter_ids = get_letter_token_ids(tokenizer, n).to(device)
    letter_logits = logits[letter_ids]
    probs = torch.softmax(letter_logits.float(), dim=0)

    ranked = torch.argsort(probs, descending=True).tolist()
    return ranked, probs.tolist()


#  CLI
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--json", type=Path, default=ROOT / "test_ranking_prompts.json")
    p.add_argument("--csv", type=Path, default=ROOT / "test_ranking.csv")
    p.add_argument(
        "--model", type=str, default="unsloth/Llama-3.2-1B-Instruct",
        help="HF model id or local PEFT/LoRA folder",
    )
    p.add_argument(
        "--adapter", type=Path, default=None,
        help="Adapter checkpoint (default: checkpoints/adapter.pt)",
    )
    p.add_argument(
        "--target-layer", type=int, default=8,
        help="Decoder layer for hidden-state addition (0-indexed)",
    )
    p.add_argument(
        "--alpha", type=float, default=0.1,
        help="Scaling factor for injection: h += alpha * z",
    )
    p.add_argument("--chat-template", action="store_true")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--n-history", type=int, default=10)
    p.add_argument("--n-candidates", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output-dir", type=Path, default=ROOT)
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
        args.max_samples = args.max_samples or 10

    device = pick_device()
    dtype = _dtype_for_device(device)
    ckpt_dir = args.checkpoint_dir

    print(f"[device] {device}  dtype={dtype}")
    print(f"[config] layer={args.target_layer}  alpha={args.alpha}")
    print(f"[model] {args.model}")

    # Load test examples and trim down the candidate set if needed.
    id_maps = IdMaps.from_json(ckpt_dir / "id_maps.json")
    examples = load_ranking_examples(args.json, args.csv, id_maps, n_history=args.n_history)
    if args.n_candidates < 10:
        examples = apply_n_candidates(examples, args.n_candidates, seed=args.seed)
    if args.max_samples:
        examples = examples[: args.max_samples]
    print(f"[data] {len(examples)} test examples, {args.n_candidates} candidates each")

    # Load the language model.
    print("[load] loading LLM...")
    llm, tokenizer = load_llm_and_tokenizer(args.model, device, dtype)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm.eval()
    for p_param in llm.parameters():
        p_param.requires_grad = False

    n_layers = llm.config.num_hidden_layers
    if args.target_layer >= n_layers:
        raise ValueError(f"--target-layer {args.target_layer} >= {n_layers} layers")
    print(f"[llm] ready  layers={n_layers}  hidden={llm.config.hidden_size}")

    # Load the adapter that maps SASRec embeddings into the LLM's hidden space.
    adapter_path = args.adapter or (ckpt_dir / "adapter.pt")
    print(f"[adapter] loading {adapter_path.name}...")
    adapter_ckpt = torch.load(adapter_path, map_location="cpu", weights_only=False)
    cfg = adapter_ckpt.get("config", {})
    adapter = EmbeddingAdapter(
        AdapterConfig(
            sasrec_dim=cfg.get("sasrec_dim", 50),
            llm_dim=cfg.get("llm_dim", llm.config.hidden_size),
            hidden_dim=cfg.get("hidden_dim", 1024),
        )
    )
    state = adapter_ckpt.get("model_state_dict") or adapter_ckpt.get("adapter_state_dict")
    if state:
        adapter.load_state_dict(state)
    adapter.to(device).eval()
    print(f"[adapter] {cfg.get('training_method', 'unknown')}  dim={cfg.get('llm_dim', '?')}")

    # Load the precomputed item embeddings from SASRec.
    item_emb = torch.load(
        ckpt_dir / "item_embeddings.pt", map_location="cpu", weights_only=True
    ).float()
    print(f"[data] {item_emb.size(0)} item embeddings")

    # Attach the injection hook to the chosen layer — this is what does the actual work.
    injector = HiddenStateInjector(alpha=args.alpha)
    injector.install(llm, layer_idx=args.target_layer)
    print(f"[hook] hidden-state addition at layer {args.target_layer}, alpha={args.alpha}")

    # Run evaluation over all test examples.
    ks = [k for k in (1, 3, 5, 10) if k < args.n_candidates] or [1]
    ranked_lists = []
    true_indices = []
    t0 = time.time()

    for i, ex in enumerate(examples):
        ranked, probs = score_example(
            llm, tokenizer, adapter, injector, ex,
            item_emb, id_maps, device, dtype,
            use_chat_template=args.chat_template,
        )
        true_idx = ex.true_position - 1
        ranked_lists.append(ranked)
        true_indices.append(true_idx)

        # Print a few examples at the start so we can sanity-check things look right.
        if i < 3:
            pred = ranked[0]
            print(
                f"  sample user {ex.user_id}: pred={LETTERS[pred]} "
                f"true={LETTERS[true_idx]} P={probs[pred]:.3f}"
            )
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(examples)}]...", flush=True)

    elapsed = time.time() - t0
    metrics = evaluate_ranked_lists(ranked_lists, true_indices, ks=ks)
    rand = random_baseline(len(examples), n_candidates=args.n_candidates, ks=ks, seed=args.seed)

    # Clean up the hook now that we're done.
    injector.remove()

    # Print results.
    model_label = Path(args.model).name if "/" not in str(args.model) else args.model.split("/")[-1]
    condition = f"HS-Add L{args.target_layer} a={args.alpha} ({model_label})"

    print(f"\n{'=' * 60}")
    print(f"RESULTS: {condition}")
    print(f"{'=' * 60}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print(f"  time: {elapsed:.1f}s")

    print(f"\n{'Condition':<40} ", end="")
    for k in ks:
        print(f"{'HR@'+str(k):>8}", end="")
    print(f"{'MRR':>8}")
    print("-" * (40 + 8 * (len(ks) + 1)))

    for label, m in [("Random", rand), (condition, metrics)]:
        print(f"{label:<40} ", end="")
        for k in ks:
            print(f"{m.get(f'HR@{k}', 0):>8.3f}", end="")
        print(f"{m.get('MRR', 0):>8.3f}")

    # Save results to a JSON file named after the config so runs don't overwrite each other.
    out_path = args.output_dir / f"results_hs_add_L{args.target_layer}_a{args.alpha}.json"
    out_path.write_text(json.dumps({
        "config": {
            "model": str(args.model),
            "adapter": str(adapter_path),
            "target_layer": args.target_layer,
            "alpha": args.alpha,
            "n_examples": len(examples),
            "n_candidates": args.n_candidates,
        },
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
        "random_baseline": {k: round(v, 4) for k, v in rand.items()},
    }, indent=2))
    print(f"\n[done] saved {out_path}")


if __name__ == "__main__":
    main()