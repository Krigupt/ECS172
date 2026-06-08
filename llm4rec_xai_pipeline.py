"""Ranking XAI engine for LLM4Rec.

This module exposes a small, notebook-friendly API for running token-level
attribution on the two ranking modes used in this repo:

- Mode A: text-only prompts with A-J labels
- Mode C: SASRec soft-token prompts with A-J labels

The heavy lifting is shared across Integrated Gradients, layer attribution,
attention rollout, ALTI+, and Grad-CAM. The module is designed to work with the
current repo layout and to accept a future local LoRA folder when it becomes
available.
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data import pick_device
from src.inject_llm import InjectedLlamaRanker, render_mode_a_prompt
from src.ranking_data import RankingExample

LETTERS = "ABCDEFGHIJ"


def _dtype_for_device(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def _token_strings(tokenizer: AutoTokenizer, ids: Sequence[int]) -> list[str]:
    return tokenizer.convert_ids_to_tokens(list(ids))


def _row_normalize(matrix: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    denom = matrix.sum(dim=-1, keepdim=True).clamp_min(eps)
    return matrix / denom


def _candidate_title_lines(suffix_text: str) -> list[str]:
    titles: list[str] = []
    for line in suffix_text.split("\n"):
        match = re.match(r"^\d+\.\s+(.+)$", line.strip())
        if match:
            titles.append(match.group(1))
    return titles


def _prompt_segment_spans_mode_a(prompt: str) -> list[tuple[str, int, int]]:
    instruction_marker = "\n\nFrom the list below, rank which movie this user would most likely enjoy:\n"
    question_marker = "\n\nReply with just the letter ("

    instruction_start = prompt.find(instruction_marker)
    question_start = prompt.find(question_marker)
    if instruction_start == -1 or question_start == -1:
        return [("prompt", 0, len(prompt))]

    candidate_start = instruction_start + len(instruction_marker)
    return [
        ("history", 0, instruction_start),
        ("instruction", instruction_start, candidate_start),
        ("candidates", candidate_start, question_start),
        ("question", question_start, len(prompt)),
    ]


def _segment_summary_from_token_scores(
    token_scores: Sequence[float] | torch.Tensor,
    token_segments: Sequence[str],
) -> dict[str, dict[str, float]]:
    if isinstance(token_scores, torch.Tensor):
        values = token_scores.detach().cpu().flatten().tolist()
    else:
        values = [float(value) for value in token_scores]

    summary: dict[str, dict[str, float]] = {}
    for score, segment in zip(values, token_segments):
        item = summary.setdefault(segment, {"signed": 0.0, "abs": 0.0, "tokens": 0.0})
        item["signed"] += float(score)
        item["abs"] += abs(float(score))
        item["tokens"] += 1.0

    total_abs = sum(item["abs"] for item in summary.values())
    if total_abs <= 0.0:
        total_abs = 1.0

    for item in summary.values():
        token_count = item["tokens"] if item["tokens"] > 0 else 1.0
        item["mean_abs"] = item["abs"] / token_count
        item["share_abs"] = item["abs"] / total_abs
    return summary


def _attention_rollout(attention_maps: Sequence[torch.Tensor]) -> torch.Tensor:
    if not attention_maps:
        raise ValueError("At least one attention map is required for rollout.")

    rollout: torch.Tensor | None = None
    for attention in attention_maps:
        seq_len = attention.size(-1)
        residual = torch.eye(seq_len, device=attention.device, dtype=attention.dtype)
        layer_flow = _row_normalize(attention + residual)
        rollout = layer_flow if rollout is None else layer_flow @ rollout

    if rollout is None:
        raise RuntimeError("Attention rollout could not be computed.")
    return _row_normalize(rollout)


def _get_transformer_layers(model: torch.nn.Module) -> Sequence[torch.nn.Module]:
    base = getattr(model, "model", None)
    if base is not None and hasattr(base, "layers"):
        return base.layers

    base_model = getattr(model, "base_model", None)
    if base_model is not None:
        inner = getattr(base_model, "model", None)
        if inner is not None and hasattr(inner, "layers"):
            return inner.layers

    raise AttributeError("Could not locate transformer layers on the loaded model.")


def _layer_output_to_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    if isinstance(output, list):
        return output[0]
    if hasattr(output, "to_tuple"):
        return output.to_tuple()[0]
    if isinstance(output, torch.Tensor):
        return output
    raise TypeError(f"Unsupported layer output type: {type(output)!r}")


def _replace_layer_output(output: Any, replacement: torch.Tensor) -> Any:
    if isinstance(output, tuple):
        return (replacement,) + tuple(output[1:])
    if isinstance(output, list):
        return [replacement] + list(output[1:])
    if hasattr(output, "to_tuple"):
        output_tuple = output.to_tuple()
        return (replacement,) + tuple(output_tuple[1:])
    return replacement


def _load_llm_and_tokenizer_eager(model_path: str | Path, device: torch.device, dtype: torch.dtype):
    """LoRA-aware loader that favors eager attention for attribution stability."""
    path = Path(model_path)
    if path.is_dir() and (path / "adapter_config.json").exists():
        cfg = json.loads((path / "adapter_config.json").read_text())
        base_name = cfg.get("base_model_name_or_path", "unsloth/Llama-3.2-1B-Instruct")
        print(f"[load] PEFT adapter from {path}")
        print(f"[load] base model {base_name} (download on first run if needed)")
        try:
            tokenizer = AutoTokenizer.from_pretrained(path)
        except (ValueError, OSError):
            tokenizer = AutoTokenizer.from_pretrained(base_name)
        base = AutoModelForCausalLM.from_pretrained(
            base_name,
            torch_dtype=dtype,
            attn_implementation="eager",
        )
        from peft import PeftModel

        llm = PeftModel.from_pretrained(base, str(path))
        llm = llm.to(device)
        print("[load] LLM ready (PEFT)")
        return llm, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    return llm, tokenizer


@contextmanager
def _patched_loader():
    import src.inject_llm as inject_llm_module

    original = inject_llm_module.load_llm_and_tokenizer
    inject_llm_module.load_llm_and_tokenizer = _load_llm_and_tokenizer_eager
    try:
        yield
    finally:
        inject_llm_module.load_llm_and_tokenizer = original


@dataclass
class SequenceBundle:
    prompt: str
    token_labels: list[str]
    token_segments: list[str]
    input_embeds: torch.Tensor
    attention_mask: torch.Tensor


def _build_mode_a_sequence(ranker: InjectedLlamaRanker, example: RankingExample) -> SequenceBundle:
    prompt = render_mode_a_prompt(example)
    encoding = ranker.tokenizer(
        prompt,
        add_special_tokens=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    input_ids = encoding["input_ids"][0].to(ranker.device)
    input_embeds = ranker._embed_tokens(input_ids)
    attention_mask = torch.ones((1, input_embeds.size(0)), dtype=torch.long, device=ranker.device)
    token_labels = _token_strings(ranker.tokenizer, input_ids.tolist())

    offsets = encoding.get("offset_mapping")
    if hasattr(offsets, "tolist"):
        offsets = offsets.tolist()
    # Hugging Face returns offsets with a batch dimension when return_tensors="pt"
    # is used. Normalize to a flat per-token list of (start, end) pairs.
    if offsets and len(offsets) == 1 and offsets[0] and isinstance(offsets[0][0], (list, tuple)):
        offsets = offsets[0]
    if offsets is None:
        token_segments = ["prompt"] * len(token_labels)
    else:
        spans = _prompt_segment_spans_mode_a(prompt)
        token_segments = []
        for index, (start, end) in enumerate(offsets):
            if start == 0 and end == 0 and index == 0:
                token_segments.append(spans[0][0] if spans else "prompt")
                continue
            segment = "other"
            for name, seg_start, seg_end in spans:
                if start < seg_end and end > seg_start:
                    segment = name
                    break
            token_segments.append(segment)

    return SequenceBundle(
        prompt=prompt,
        token_labels=token_labels,
        token_segments=token_segments,
        input_embeds=input_embeds.unsqueeze(0),
        attention_mask=attention_mask,
    )


def _build_mode_c_sequence(ranker: InjectedLlamaRanker, example: RankingExample) -> SequenceBundle:
    n = len(example.candidate_movie_ids)
    letter_range = f"{LETTERS[0]}-{LETTERS[n - 1]}"
    prompt = (
        example.prefix_text
        + "\n\nEach candidate below is represented as a collaborative filtering "
        + "embedding that encodes viewing patterns from similar users:\n"
        + "\n".join(
            f"{LETTERS[i]}. [SOFT_TOKEN movie_id={mid}]"
            for i, mid in enumerate(example.candidate_movie_ids)
        )
        + f"\n\nReply with just the letter ({letter_range}) of the movie "
        + "they would rate highest.\nAnswer:"
    )

    prefix_ids = ranker.tokenizer(
        example.prefix_text,
        add_special_tokens=True,
        return_tensors="pt",
    ).input_ids[0]
    parts: list[torch.Tensor] = [ranker._embed_tokens(prefix_ids.to(ranker.device))]
    labels: list[str] = ranker.tokenizer.convert_ids_to_tokens(prefix_ids.tolist())
    segments: list[str] = ["history"] * len(labels)

    framing = (
        "\n\nEach candidate below is represented as a collaborative filtering "
        "embedding that encodes viewing patterns from similar users:"
    )
    framing_ids = ranker._tokenize_text(framing)
    parts.append(ranker._embed_tokens(framing_ids.to(ranker.device)))
    labels.extend(ranker.tokenizer.convert_ids_to_tokens(framing_ids.tolist()))
    segments.extend(["instruction"] * framing_ids.size(0))

    candidate_vectors = ranker.candidate_vectors(example.candidate_movie_ids)
    for index, (mid, vec) in enumerate(zip(example.candidate_movie_ids, candidate_vectors, strict=True)):
        bullet_ids = ranker._tokenize_text(f"\n{LETTERS[index]}. ")
        parts.append(ranker._embed_tokens(bullet_ids.to(ranker.device)))
        labels.extend(ranker.tokenizer.convert_ids_to_tokens(bullet_ids.tolist()))
        segments.extend(["candidates"] * bullet_ids.size(0))

        parts.append(vec)
        labels.append(f"[SOFT_TOKEN movie_id={int(mid)}]")
        segments.append("candidates")

    footer_ids = ranker._tokenize_text(
        f"\n\nReply with just the letter ({letter_range}) of the movie "
        "they would rate highest.\nAnswer:"
    )
    parts.append(ranker._embed_tokens(footer_ids.to(ranker.device)))
    labels.extend(ranker.tokenizer.convert_ids_to_tokens(footer_ids.tolist()))
    segments.extend(["question"] * footer_ids.size(0))

    input_embeds = torch.cat(parts, dim=0).unsqueeze(0)
    attention_mask = torch.ones((1, input_embeds.size(1)), dtype=torch.long, device=ranker.device)
    return SequenceBundle(
        prompt=prompt,
        token_labels=labels,
        token_segments=segments,
        input_embeds=input_embeds,
        attention_mask=attention_mask,
    )


def _make_forward_score_from_embeds(
    ranker: InjectedLlamaRanker,
    target_continuation: str,
    comparison_continuations: Sequence[str] | None = None,
):
    target_ids = ranker.tokenizer.encode(target_continuation, add_special_tokens=False)
    if len(target_ids) != 1:
        raise RuntimeError(
            f"Target continuation {target_continuation!r} should be a single token, got {target_ids}."
        )
    target_id = target_ids[0]

    comparison_ids: list[int] = []
    for continuation in comparison_continuations or []:
        if continuation == target_continuation:
            continue
        ids = ranker.tokenizer.encode(continuation, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(
                f"Comparison continuation {continuation!r} should be a single token, got {ids}."
            )
        comparison_ids.append(ids[0])

    def _forward(inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = ranker.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        next_token_log_probs = F.log_softmax(outputs.logits[:, -1, :].float(), dim=-1)
        target = next_token_log_probs[:, target_id]
        if not comparison_ids:
            return target
        comparison_scores = [next_token_log_probs[:, idx] for idx in comparison_ids]
        return target - torch.stack(comparison_scores, dim=0).max(dim=0).values

    return _forward


def _score_letter_distribution(
    ranker: InjectedLlamaRanker,
    sequence: SequenceBundle,
    n_candidates: int,
) -> tuple[list[dict[str, Any]], list[int], list[float], list[float]]:
    letter_ids = ranker._get_letter_token_ids(n_candidates)
    outputs = ranker.llm(
        inputs_embeds=sequence.input_embeds,
        attention_mask=sequence.attention_mask,
        use_cache=False,
        return_dict=True,
    )
    next_token_log_probs = F.log_softmax(outputs.logits[0, -1, :].float(), dim=-1)
    letter_log_probs = next_token_log_probs[letter_ids]
    probs = torch.softmax(letter_log_probs.float(), dim=0)

    scored: list[dict[str, Any]] = []
    for index in range(n_candidates):
        label = LETTERS[index]
        positive = letter_log_probs[index]
        other_values = [letter_log_probs[j] for j in range(n_candidates) if j != index]
        negative = torch.stack(other_values, dim=0).max(dim=0).values if other_values else positive
        score = {
            "score_margin": float((positive - negative).item()),
            "positive_logprob": float(positive.item()),
            "negative_logprob": float(negative.item()),
        }
        scored.append(
            {
                "prompt_label": label,
                "score": score,
                "prob": float(probs[index].item()),
            }
        )

    ranked_indices = torch.argsort(probs, descending=True).tolist()
    return scored, ranked_indices, probs.tolist(), letter_log_probs.tolist()


def _integrated_gradients(
    ranker: InjectedLlamaRanker,
    sequence: SequenceBundle,
    target_continuation: str,
    comparison_continuations: Sequence[str] | None,
    n_steps: int = 32,
    internal_batch_size: int | None = 1,
) -> dict[str, Any]:
    forward_func = _make_forward_score_from_embeds(
        ranker,
        target_continuation=target_continuation,
        comparison_continuations=comparison_continuations,
    )
    ig = IntegratedGradients(forward_func)
    baseline = torch.zeros_like(sequence.input_embeds)
    attributions, delta = ig.attribute(
        inputs=sequence.input_embeds,
        baselines=baseline,
        additional_forward_args=(sequence.attention_mask,),
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
        return_convergence_delta=True,
    )
    token_scores = attributions.sum(dim=-1).squeeze(0)
    token_scores_abs = attributions.abs().sum(dim=-1).squeeze(0)
    segment_scores = _segment_summary_from_token_scores(token_scores, sequence.token_segments)
    return {
        "tokens": list(sequence.token_labels),
        "token_scores": token_scores.detach().cpu().tolist(),
        "token_scores_abs": token_scores_abs.detach().cpu().tolist(),
        "segment_scores": segment_scores,
        "convergence_delta": float(delta.item()),
    }


def _layer_attributions(
    ranker: InjectedLlamaRanker,
    sequence: SequenceBundle,
    target_continuation: str,
    comparison_continuations: Sequence[str] | None,
    layer_indices: Sequence[int] | None,
    attribution_method: str,
    n_steps: int = 32,
    internal_batch_size: int | None = 1,
) -> list[dict[str, Any]]:
    forward_func = _make_forward_score_from_embeds(
        ranker,
        target_continuation=target_continuation,
        comparison_continuations=comparison_continuations,
    )
    layers = _get_transformer_layers(ranker.llm)
    if layer_indices is None:
        layer_indices = list(range(len(layers)))

    method = attribution_method.lower().strip()
    results: list[dict[str, Any]] = []
    tokens = list(sequence.token_labels)

    target_ids = ranker.tokenizer.encode(target_continuation, add_special_tokens=False)
    if len(target_ids) != 1:
        raise RuntimeError(
            f"Target continuation {target_continuation!r} should be a single token, got {target_ids}."
        )
    target_id = target_ids[0]

    comparison_ids: list[int] = []
    for continuation in comparison_continuations or []:
        if continuation == target_continuation:
            continue
        ids = ranker.tokenizer.encode(continuation, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(
                f"Comparison continuation {continuation!r} should be a single token, got {ids}."
            )
        comparison_ids.append(ids[0])

    def _score_input_embeds() -> dict[str, float]:
        outputs = ranker.llm(
            inputs_embeds=sequence.input_embeds,
            attention_mask=sequence.attention_mask,
            use_cache=False,
            return_dict=True,
        )
        next_token_log_probs = F.log_softmax(outputs.logits[:, -1, :].float(), dim=-1)
        positive = next_token_log_probs[:, target_id]
        if not comparison_ids:
            negative = positive
        else:
            negative = torch.stack([next_token_log_probs[:, idx] for idx in comparison_ids], dim=0).max(dim=0).values
        return {
            "score_margin": float((positive - negative).item()),
            "positive_logprob": float(positive.item()),
            "negative_logprob": float(negative.item()),
        }

    for layer_index in layer_indices:
        layer = layers[layer_index]
        attribution_mode = "layer_output"
        delta_value: float | None = None

        if method in {"feature_ablation", "fa", "layer_feature_ablation"}:
            base_score = _score_input_embeds()
            capture: dict[str, torch.Tensor] = {}

            def _hook(_module: torch.nn.Module, inputs: tuple[Any, ...], output: Any) -> Any:
                layer_input = inputs[0]
                layer_output = _layer_output_to_tensor(output)
                capture["layer_input"] = layer_input.detach()
                capture["layer_output"] = layer_output.detach()
                return _replace_layer_output(output, layer_input)

            handle = layer.register_forward_hook(_hook)
            try:
                ablated_score = _score_input_embeds()
            finally:
                handle.remove()

            if "layer_input" not in capture or "layer_output" not in capture:
                raise RuntimeError(
                    f"Failed to capture activations for layer {layer_index} during causal layer ablation."
                )

            delta_hidden = capture["layer_output"] - capture["layer_input"]
            score_delta = float(base_score["score_margin"] - ablated_score["score_margin"])
            token_scores = delta_hidden.mean(dim=-1).squeeze(0)
            token_scores_abs = delta_hidden.abs().mean(dim=-1).squeeze(0)
            segment_scores = _segment_summary_from_token_scores(token_scores, sequence.token_segments)
            results.append(
                {
                    "layer_index": layer_index,
                    "attribution_method": "causal_bypass",
                    "tokens": tokens,
                    "token_scores": token_scores.detach().cpu().tolist(),
                    "token_scores_abs": token_scores_abs.detach().cpu().tolist(),
                    "layer_score": abs(score_delta),
                    "segment_scores": segment_scores,
                    "attribution_mode": "causal_bypass",
                    "convergence_delta": score_delta,
                }
            )
            continue

        def _attribute_with_mode(attribute_to_layer_input: bool):
            if method in {"lig", "layer_integrated_gradients", "integrated_gradients"}:
                explainer = LayerIntegratedGradients(forward_func, layer)
                return explainer.attribute(
                    inputs=sequence.input_embeds,
                    baselines=torch.zeros_like(sequence.input_embeds),
                    additional_forward_args=(sequence.attention_mask,),
                    n_steps=n_steps,
                    internal_batch_size=internal_batch_size,
                    attribute_to_layer_input=attribute_to_layer_input,
                    return_convergence_delta=True,
                )
            raise ValueError(
                f"Unsupported layer attribution method: {attribution_method!r}. Use 'lig' or 'feature_ablation'."
            )

        try:
            attr, delta = _attribute_with_mode(False)
        except Exception:
            attr, delta = _attribute_with_mode(True)
            attribution_mode = "layer_input"

        layer_score_tensor = attr.abs().sum()
        if float(layer_score_tensor.item()) <= 0.0 and attribution_mode == "layer_output":
            try:
                attr_input, delta_input = _attribute_with_mode(True)
                input_score_tensor = attr_input.abs().sum()
                if float(input_score_tensor.item()) > float(layer_score_tensor.item()):
                    attr = attr_input
                    delta = delta_input
                    attribution_mode = "layer_input"
            except Exception:
                pass

        if isinstance(delta, torch.Tensor) and torch.isfinite(delta).all():
            delta_value = float(delta.item())

        token_scores = attr.mean(dim=-1).squeeze(0)
        token_scores_abs = attr.abs().mean(dim=-1).squeeze(0)
        segment_scores = _segment_summary_from_token_scores(token_scores, sequence.token_segments)
        results.append(
            {
                "layer_index": layer_index,
                "attribution_method": method,
                "tokens": tokens,
                "token_scores": token_scores.detach().cpu().tolist(),
                "token_scores_abs": token_scores_abs.detach().cpu().tolist(),
                "layer_score": float(token_scores_abs.sum().item()),
                "segment_scores": segment_scores,
                "attribution_mode": attribution_mode,
                "convergence_delta": delta_value,
            }
        )

    return results


def _attention_maps(
    ranker: InjectedLlamaRanker,
    sequence: SequenceBundle,
    layer_indices: Sequence[int] | None,
) -> dict[str, Any]:
    outputs = ranker.llm(
        inputs_embeds=sequence.input_embeds,
        attention_mask=sequence.attention_mask,
        output_attentions=True,
        use_cache=False,
        return_dict=True,
    )
    attentions = list(outputs.attentions or [])
    if not attentions:
        raise RuntimeError("The model did not return any attention maps.")

    if layer_indices is None:
        layer_indices = list(range(len(attentions)))
    else:
        layer_indices = list(layer_indices)

    raw_attention_maps: list[torch.Tensor] = []
    layer_target_rows: list[torch.Tensor] = []
    for layer_index in layer_indices:
        attn = attentions[layer_index].mean(dim=1)[0].detach().cpu()
        raw_attention_maps.append(attn)

    rollout = _attention_rollout([attentions[layer_index].mean(dim=1)[0] for layer_index in layer_indices]).detach().cpu()
    target_index = sequence.input_embeds.size(1) - 1
    for attn in raw_attention_maps:
        layer_target_rows.append(attn[target_index].detach().cpu())

    target_row = rollout[target_index]
    segment_scores = _segment_summary_from_token_scores(target_row, sequence.token_segments)
    return {
        "tokens": list(sequence.token_labels),
        "layer_indices": layer_indices,
        "target_index": target_index,
        "layer_attention_maps": [attn.tolist() for attn in raw_attention_maps],
        "layer_target_rows": [row.tolist() for row in layer_target_rows],
        "rollout": rollout.tolist(),
        "target_row": target_row.tolist(),
        "segment_scores": segment_scores,
    }


def _alti_plus(
    ranker: InjectedLlamaRanker,
    sequence: SequenceBundle,
) -> dict[str, Any]:
    outputs = ranker.llm(
        inputs_embeds=sequence.input_embeds,
        attention_mask=sequence.attention_mask,
        output_attentions=True,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

    hidden_states = list(outputs.hidden_states or [])
    attentions = list(outputs.attentions or [])
    layers = _get_transformer_layers(ranker.llm)
    if not hidden_states or not attentions:
        raise RuntimeError("The model did not return hidden states / attentions for ALTI+.")

    num_heads = getattr(ranker.llm.config, "num_attention_heads", None)
    if num_heads is None:
        num_heads = getattr(ranker.llm.config, "num_heads", None)
    if num_heads is None:
        raise AttributeError("Could not determine attention head count for ALTI+.")

    num_kv_heads = getattr(ranker.llm.config, "num_key_value_heads", None)
    if num_kv_heads is None:
        num_kv_heads = num_heads

    head_dim = getattr(ranker.llm.config, "head_dim", None)
    if head_dim is None:
        head_dim = hidden_states[0].shape[-1] // num_heads

    layer_results: list[dict[str, Any]] = []
    matrices: list[torch.Tensor] = []

    for layer_index, layer in enumerate(layers):
        hidden = hidden_states[layer_index]
        attn = attentions[layer_index][0]
        x_norm = layer.input_layernorm(hidden)
        v = layer.self_attn.v_proj(x_norm)

        batch, seq_len, _ = v.shape
        repeat_factor = max(1, num_heads // num_kv_heads)
        v = v.view(batch, seq_len, num_kv_heads, head_dim)
        if repeat_factor > 1:
            v = v.repeat_interleave(repeat_factor, dim=2)
        v = v[0]

        rows: list[torch.Tensor] = []
        for dest_idx in range(seq_len):
            source_head_contrib = v * attn[:, dest_idx, :].transpose(0, 1).unsqueeze(-1)
            source_vectors = layer.self_attn.o_proj(source_head_contrib.reshape(seq_len, -1))
            source_vectors_with_residual = source_vectors.clone()
            source_vectors_with_residual[dest_idx] = source_vectors_with_residual[dest_idx] + hidden[0, dest_idx]

            layer_output = hidden[0, dest_idx] + source_vectors.sum(dim=0)
            layer_output_l1 = layer_output.abs().sum()
            distances = (layer_output.unsqueeze(0) - source_vectors_with_residual).abs().sum(dim=-1)
            contributions = torch.relu(layer_output_l1 - distances)

            if float(contributions.sum().item()) <= 0.0:
                row = torch.zeros_like(contributions)
                row[dest_idx] = 1.0
            else:
                row = contributions / contributions.sum()
            rows.append(row)

        matrix = torch.stack(rows, dim=0)
        matrices.append(matrix)
        matrix_cpu = matrix.detach().cpu()
        layer_results.append(
            {
                "layer_index": layer_index,
                "matrix": matrix_cpu.tolist(),
                "target_row": matrix_cpu[-1].tolist(),
                "target_row_abs_sum": float(matrix_cpu[-1].abs().sum().item()),
            }
        )

    rollout = matrices[0]
    for matrix in matrices[1:]:
        rollout = matrix @ rollout
    rollout = _row_normalize(rollout)

    target_index = sequence.input_embeds.size(1) - 1
    target_row = rollout[target_index]
    segment_scores = _segment_summary_from_token_scores(target_row, sequence.token_segments)
    return {
        "tokens": list(sequence.token_labels),
        "target_index": target_index,
        "layer_results": layer_results,
        "rollout": rollout.tolist(),
        "target_row": target_row.tolist(),
        "segment_scores": segment_scores,
    }


def _grad_cam(
    ranker: InjectedLlamaRanker,
    sequence: SequenceBundle,
    target_continuation: str,
    negative_continuation: str,
) -> dict[str, Any]:
    def _cam_for_continuation(continuation: str) -> list[torch.Tensor]:
        full_embeds = sequence.input_embeds.detach().requires_grad_(True)
        outputs = ranker.llm(
            inputs_embeds=full_embeds,
            attention_mask=sequence.attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        attentions = list(outputs.attentions or [])
        hidden_states = list(outputs.hidden_states or [])
        for hidden_state in hidden_states:
            if hidden_state.requires_grad:
                hidden_state.retain_grad()

        continuation_ids = ranker.tokenizer.encode(continuation, add_special_tokens=False)
        if len(continuation_ids) != 1:
            raise RuntimeError("Grad-CAM continuation labels must each be a single token.")

        next_token_log_probs = F.log_softmax(outputs.logits[:, -1, :].float(), dim=-1)
        score = next_token_log_probs[:, continuation_ids[0]]
        score.backward()

        cams_local: list[torch.Tensor] = []
        for layer_index, attn in enumerate(attentions):
            attn_map = attn.mean(dim=1)[0]
            hidden_state = (
                hidden_states[layer_index + 1]
                if layer_index + 1 < len(hidden_states)
                else hidden_states[layer_index]
            )
            if hidden_state is not None and hidden_state.grad is not None:
                token_importance = hidden_state.grad.abs().sum(dim=-1)[0]
            else:
                token_importance = torch.relu(attn_map.sum(dim=0))
            cam = torch.relu(attn_map * token_importance.unsqueeze(0))
            cam = _row_normalize(cam.clamp_min(0.0))
            cams_local.append(cam.detach().cpu())
        return cams_local

    pos_cams = _cam_for_continuation(target_continuation)
    neg_cams = _cam_for_continuation(negative_continuation)
    margin_cams = [pos - neg for pos, neg in zip(pos_cams, neg_cams, strict=True)]
    pos_target_row = pos_cams[-1][-1] if pos_cams else torch.zeros(len(sequence.token_labels), dtype=torch.float32)
    neg_target_row = neg_cams[-1][-1] if neg_cams else torch.zeros(len(sequence.token_labels), dtype=torch.float32)
    margin_target_row = margin_cams[-1][-1] if margin_cams else torch.zeros(len(sequence.token_labels), dtype=torch.float32)
    return {
        "prompt_len": sequence.input_embeds.size(1),
        "positive": {
            "tokens": list(sequence.token_labels),
            "maps": [cam.tolist() for cam in pos_cams],
            "target_rows": pos_target_row.tolist() if pos_cams else [],
            "segment_scores": _segment_summary_from_token_scores(pos_target_row, sequence.token_segments),
        },
        "negative": {
            "tokens": list(sequence.token_labels),
            "maps": [cam.tolist() for cam in neg_cams],
            "target_rows": neg_target_row.tolist() if neg_cams else [],
            "segment_scores": _segment_summary_from_token_scores(neg_target_row, sequence.token_segments),
        },
        "margin": {
            "tokens": list(sequence.token_labels),
            "maps": [cam.tolist() for cam in margin_cams],
            "target_rows": margin_target_row.tolist() if margin_cams else [],
            "segment_scores": _segment_summary_from_token_scores(margin_target_row, sequence.token_segments),
        },
    }


def _segment_share_percentages(segment_scores: dict[str, dict[str, float]]) -> dict[str, float]:
    return {segment: float(values.get("share_abs", 0.0)) * 100.0 for segment, values in segment_scores.items()}


class RankingXAIPipeline:
    """Frozen Llama + ranking XAI for the current repo layout."""

    def __init__(
        self,
        model_name: str | Path,
        checkpoint_dir: str | Path = "checkpoints",
        device: torch.device | None = None,
        freeze_llm: bool = True,
        train_adapter: bool = False,
        load_embedding_adapter: bool = True,
        embedding_adapter_path: str | Path | None = None,
        projected_embeddings_path: str | Path | None = None,
    ) -> None:
        self.device = device or pick_device()
        self.checkpoint_dir = Path(checkpoint_dir)
        with _patched_loader():
            self.ranker = InjectedLlamaRanker(
                model_name=model_name,
                checkpoint_dir=self.checkpoint_dir,
                device=self.device,
                freeze_llm=freeze_llm,
                train_adapter=train_adapter,
                load_embedding_adapter=load_embedding_adapter,
                embedding_adapter_path=embedding_adapter_path,
                projected_embeddings_path=projected_embeddings_path,
            )
        self.model = self.ranker.llm
        self.tokenizer = self.ranker.tokenizer

    def build_sequence(self, example: RankingExample, mode: str = "text") -> SequenceBundle:
        if mode == "text":
            return _build_mode_a_sequence(self.ranker, example)
        if mode == "candidates":
            return _build_mode_c_sequence(self.ranker, example)
        raise ValueError(f"Unknown mode {mode!r}; expected 'text' or 'candidates'.")

    def rank_example(self, example: RankingExample, mode: str = "text") -> tuple[list[dict[str, Any]], SequenceBundle]:
        sequence = self.build_sequence(example, mode=mode)
        scored, ranked_indices, probs, letter_log_probs = _score_letter_distribution(
            self.ranker,
            sequence,
            len(example.candidate_movie_ids),
        )
        ranked = [
            {
                "candidate": {"movie_id": int(example.candidate_movie_ids[idx])},
                "prompt": sequence.prompt,
                "prompt_label": LETTERS[idx],
                "score": next(item["score"] for item in scored if item["prompt_label"] == LETTERS[idx]),
            }
            for idx in ranked_indices
        ]
        return ranked, sequence

    def analyze_example(
        self,
        example: RankingExample,
        mode: str = "text",
        target_mode: str = "model_choice",
        layer_indices: Sequence[int] | None = None,
        ig_steps: int = 32,
        layer_steps: int = 16,
        layer_attribution_method: str = "lig",
        include_attention_maps: bool = False,
        include_metrics: bool = False,
    ) -> dict[str, Any]:
        """Run a single XAI pass.

        Metrics are optional and disabled by default so notebook runs stay
        focused on explanation artifacts rather than re-evaluating ranking.
        """
        sequence = self.build_sequence(example, mode=mode)
        scored, ranked_indices, probs, _letter_log_probs = _score_letter_distribution(
            self.ranker,
            sequence,
            len(example.candidate_movie_ids),
        )

        ranked = [
            {
                "candidate": {"movie_id": int(example.candidate_movie_ids[idx])},
                "prompt": sequence.prompt,
                "prompt_label": LETTERS[idx],
                "score": next(item["score"] for item in scored if item["prompt_label"] == LETTERS[idx]),
            }
            for idx in ranked_indices
        ]

        if target_mode == "ground_truth":
            target_label = LETTERS[example.true_position - 1]
        elif target_mode == "model_choice":
            target_label = str(ranked[0]["prompt_label"])
        else:
            raise ValueError(f"Unknown target_mode={target_mode!r}")

        comparison_label = None
        for item in ranked:
            if item["prompt_label"] != target_label:
                comparison_label = str(item["prompt_label"])
                break
        comparison_continuations = [f" {comparison_label}"] if comparison_label is not None else []
        target_continuation = f" {target_label}"

        target_item = next(item for item in ranked if item["prompt_label"] == target_label)
        selection = {
            "target_candidate": target_item["candidate"],
            "target_candidate_id": int(example.candidate_movie_ids[LETTERS.index(target_label)]),
            "target_candidate_rank": ranked_indices.index(LETTERS.index(target_label)) + 1,
            "target_label": target_label,
            "target_score": target_item["score"],
            "positive_candidate": {"movie_id": int(example.true_positive_movie_id)},
            "positive_candidate_id": int(example.true_positive_movie_id),
            "positive_candidate_rank": ranked_indices.index(example.true_position - 1) + 1,
            "top_candidate": ranked[0]["candidate"],
            "top_candidate_id": int(example.candidate_movie_ids[ranked_indices[0]]),
            "top_candidate_rank": 1,
            "top_candidate_label": ranked[0]["prompt_label"],
            "top_score": ranked[0]["score"],
        }

        xai_result = {
            "score": target_item["score"],
            "integrated_gradients": _integrated_gradients(
                self.ranker,
                sequence,
                target_continuation=target_continuation,
                comparison_continuations=comparison_continuations,
                n_steps=ig_steps,
            ),
            "layer_attributions": _layer_attributions(
                self.ranker,
                sequence,
                target_continuation=target_continuation,
                comparison_continuations=comparison_continuations,
                layer_indices=layer_indices,
                attribution_method=layer_attribution_method,
                n_steps=layer_steps,
            ),
            "attention_maps": _attention_maps(self.ranker, sequence, layer_indices=layer_indices) if include_attention_maps else None,
            "alti_plus": _alti_plus(self.ranker, sequence),
            "grad_cam": _grad_cam(
                self.ranker,
                sequence,
                target_continuation=target_continuation,
                negative_continuation=f" {comparison_label or target_label}",
            ),
        }

        segment_summary = {
            "integrated_gradients": _segment_share_percentages(xai_result["integrated_gradients"]["segment_scores"]),
            "grad_cam": _segment_share_percentages(xai_result["grad_cam"]["margin"]["segment_scores"]),
            "alti_plus": _segment_share_percentages(xai_result["alti_plus"]["segment_scores"]),
        }
        if include_attention_maps and xai_result["attention_maps"] is not None:
            segment_summary["attention_rollout"] = _segment_share_percentages(xai_result["attention_maps"]["segment_scores"])

        result: dict[str, Any] = {
            "model_id": str(self.ranker.model_path),
            "example_id": int(example.user_id),
            "split": "test",
            "ranking_mode": mode,
            "target_mode": target_mode,
            "prompt": sequence.prompt,
            "prompt_tokens": list(sequence.token_labels),
            "prompt_token_count": len(sequence.token_labels),
            "ranked_candidates": ranked,
            "selection": selection,
            "segment_summary": segment_summary,
            "layer_attributions": xai_result["layer_attributions"],
            "xai_result": xai_result,
        }
        if include_metrics:
            from src.metrics import evaluate_ranked_lists

            result["metrics"] = evaluate_ranked_lists([ranked_indices], [example.true_position - 1], ks=(1, 3, 5))
        return result


def available_configs(
    root: Path,
    checkpoint_dir: Path,
    base_model: str = "unsloth/Llama-3.2-1B-Instruct",
    lora_model: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Convenience helper for notebooks."""
    lora_model = Path(lora_model) if lora_model is not None else root / "llama31-1b-movielens-ranking-lora"
    return [
        {
            "key": "base",
            "label": "Base",
            "mode": "text",
            "model_name": base_model,
            "load_embedding_adapter": False,
            "embedding_adapter_path": None,
            "projected_embeddings_path": None,
        },
        {
            "key": "base_sas",
            "label": "Base + SAS",
            "mode": "candidates",
            "model_name": base_model,
            "load_embedding_adapter": True,
            "embedding_adapter_path": checkpoint_dir / "adapter_ranking.pt",
            "projected_embeddings_path": checkpoint_dir / "projected_embeddings_ranking.pt",
        },
        {
            "key": "lora",
            "label": "LoRA",
            "mode": "text",
            "model_name": lora_model,
            "load_embedding_adapter": False,
            "embedding_adapter_path": None,
            "projected_embeddings_path": None,
            "optional": True,
        },
        {
            "key": "lora_sas",
            "label": "LoRA + SAS",
            "mode": "candidates",
            "model_name": lora_model,
            "load_embedding_adapter": True,
            "embedding_adapter_path": checkpoint_dir / "adapter_ranking_lora.pt",
            "projected_embeddings_path": checkpoint_dir / "projected_embeddings_ranking_lora.pt",
            "optional": True,
        },
    ]
