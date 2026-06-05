"""Layer-level KV injection adapter (P-Tuning v2 style).

Instead of prepending a soft token at the input embedding layer (layer 0),
this adapter injects learned hidden-state vectors at a *specific* transformer
decoder layer.  The target layer's own K/V projections turn those vectors
into attention keys and values — so every real token in the sequence can
attend to the injected collaborative signal through normal self-attention.

Concretely, before layer L runs:
    1. Prefix hidden states (from the adapter) are prepended to the sequence.
    2. The attention mask & position embeddings are extended to cover them.
After layer L:
    3. The prefix positions are stripped from the output hidden states.

From the real tokens' perspective this is equivalent to KV-injection:
they see extra K/V entries produced by the layer's own k_proj/v_proj
applied to the prefix, and attend to them via normal scaled-dot-product
attention.  No monkey-patching of attention internals is needed.

References:
    - Prefix-Tuning (Li & Liang, 2021)
    - P-Tuning v2   (Liu et al., 2022)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn

@dataclass
class KVAdapterConfig:
    sasrec_dim: int = 50
    llm_dim: int = 2048
    hidden_dim: int = 1024       # intermediate MLP width
    n_prefix: int = 1            # virtual tokens per injection
    target_layer: int = 6        # decoder layer index to inject at


class KVAdapter(nn.Module):
    """Project a SASRec item embedding into prefix hidden-state(s) for
    injection at a specific transformer layer.

    Architecture (mirrors EmbeddingAdapter):
        Linear(sasrec_dim → hidden_dim) → SiLU → LayerNorm
            → Linear(hidden_dim → n_prefix * llm_dim)

    The output is reshaped to ``(batch, n_prefix, llm_dim)`` and fed to
    :class:`InjectedDecoderLayer` as prefix hidden states.
    """

    def __init__(self, config: KVAdapterConfig | None = None):
        super().__init__()
        config = config or KVAdapterConfig()
        self.config = config

        self.projector = nn.Sequential(
            nn.Linear(config.sasrec_dim, config.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.n_prefix * config.llm_dim),
        )

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """Map SASRec embedding(s) to prefix hidden states.

        Args:
            e: ``(batch, sasrec_dim)``

        Returns:
            ``(batch, n_prefix, llm_dim)``  — ready for InjectedDecoderLayer.
        """
        out = self.projector(e)                         # (B, n_prefix * llm_dim)
        B = e.size(0)
        return out.view(B, self.config.n_prefix, self.config.llm_dim)

    def project(self, e: torch.Tensor) -> torch.Tensor:
        """Alias for ``forward`` — matches EmbeddingAdapter's interface."""
        return self.forward(e)


class InjectedDecoderLayer(nn.Module):
    """Thin wrapper around a transformer decoder layer that prepends
    prefix hidden states when they are set, and strips them from the
    output so downstream layers see the original sequence length.

    Install with :func:`install_layer_injection`.

    Usage::

        wrapper = install_layer_injection(llm, target_layer=6, n_prefix=1)
        # before each forward pass:
        wrapper.set_prefix(prefix_hs)   # (B, n_prefix, D)
        outputs = llm(input_ids=..., ...)
        wrapper.clear_prefix()
    """

    def __init__(self, original_layer: nn.Module, n_prefix: int):
        super().__init__()
        self.layer = original_layer     # the real decoder layer we're wrapping
        self.n_prefix = n_prefix
        self._prefix_hs: Optional[torch.Tensor] = None

    # ── public API ────────────────────────────────────────────────

    def set_prefix(self, prefix_hs: Optional[torch.Tensor]):
        """Set prefix hidden states for the next forward pass.

        Args:
            prefix_hs: ``(batch, n_prefix, hidden_dim)`` or ``None`` to
                       disable injection (pass-through mode).
        """
        self._prefix_hs = prefix_hs

    def clear_prefix(self):
        """Remove prefix — layer reverts to normal behaviour."""
        self._prefix_hs = None

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        """Wrap the original layer: prepend prefix, forward, strip prefix."""

        # If no prefix is set, just pass through to the original layer unchanged.
        if self._prefix_hs is None:
            return self.layer(hidden_states, **kwargs)

        n = self.n_prefix
        device = hidden_states.device
        dtype = hidden_states.dtype
        batch_size = hidden_states.size(0)
        prefix = self._prefix_hs.to(device=device, dtype=dtype)

        extended_hs = torch.cat([prefix, hidden_states], dim=1)

        # Work on a copy of kwargs so we don't accidentally mutate the caller's dict.
        kw = dict(kwargs)

        mask = kw.get("attention_mask")
        if mask is not None:
            kw["attention_mask"] = _extend_mask(mask, n, device)

        # Prefix tokens get position 0 so they're treated as position-free.
        # Real tokens keep their original positions so RoPE stays consistent
        # with what every other layer sees.
        pos = kw.get("position_ids")
        if pos is not None:
            pfx_pos = torch.zeros(batch_size, n, dtype=pos.dtype, device=device)
            kw["position_ids"] = torch.cat([pfx_pos, pos], dim=1)

        # Newer versions of HF transformers pre-compute the RoPE (cos, sin) tensors
        # and pass them in directly. We replicate the position-0 embedding for the
        # prefix slots so the layer doesn't crash on a shape mismatch.
        pos_emb = kw.get("position_embeddings")
        if pos_emb is not None:
            cos, sin = pos_emb
            if cos.dim() == 3:                          # (B, seq, head_dim)
                pc = cos[:, :1, :].expand(-1, n, -1)
                ps = sin[:, :1, :].expand(-1, n, -1)
                kw["position_embeddings"] = (
                    torch.cat([pc, cos], dim=1),
                    torch.cat([ps, sin], dim=1),
                )
            elif cos.dim() == 2:                        # (seq, head_dim)
                pc = cos[:1, :].expand(n, -1)
                ps = sin[:1, :].expand(n, -1)
                kw["position_embeddings"] = (
                    torch.cat([pc, cos], dim=0),
                    torch.cat([ps, sin], dim=0),
                )

        # Run the extended sequence through the real decoder layer.
        outputs = self.layer(extended_hs, **kw)

        # Strip the prefix tokens back off so the rest of the network
        # sees the original sequence length and doesn't need to know any
        # of this happened.
        out_hs = outputs[0][:, n:, :]
        return (out_hs,) + outputs[1:]

def install_layer_injection( llm: nn.Module, target_layer: int, n_prefix: int = 1,) -> InjectedDecoderLayer:
    """Replace ``llm.model.layers[target_layer]`` with an
    :class:`InjectedDecoderLayer` wrapper and return it.

    The original layer becomes a sub-module of the wrapper, so its
    (frozen) parameters remain accessible and unchanged.

    Args:
        llm: A HuggingFace ``LlamaForCausalLM`` (or merged PEFT model).
        target_layer: Index of the decoder layer to wrap.
        n_prefix: Number of virtual prefix tokens.

    Returns:
        The installed :class:`InjectedDecoderLayer`.  Call
        ``wrapper.set_prefix(...)`` before each forward pass.
    """
    # The attribute path to the layer list differs between a plain model
    # and one that's been wrapped by PEFT/LoRA.
    if hasattr(llm, "model") and hasattr(llm.model, "layers"):
        layers = llm.model.layers
    elif hasattr(llm, "base_model"):
        layers = llm.base_model.model.layers
    else:
        raise AttributeError(
            "Cannot locate decoder layers.  Expected "
            "llm.model.layers (LlamaForCausalLM) or "
            "llm.base_model.model.layers (merged PEFT)."
        )

    original = layers[target_layer]
    wrapper = InjectedDecoderLayer(original, n_prefix)
    layers[target_layer] = wrapper

    print(
        f"[kv_adapter] installed InjectedDecoderLayer at layer {target_layer} "
        f"(n_prefix={n_prefix})",
        flush=True,
    )
    return wrapper

def _extend_mask( mask: torch.Tensor, n_prefix: int, device: torch.device,) -> torch.Tensor:
    n = n_prefix
    if mask.dim() == 4:
        B, H, Q, KV = mask.shape
        # Build a new mask that covers the prefix slots too. The bottom-right
        # block stays as the original causal mask; everything else is 0 (allowed)
        # so real tokens can freely attend to the prefix.
        ext = torch.zeros(B, H, n + Q, n + KV, device=device, dtype=mask.dtype)
        ext[:, :, n:, n:] = mask
        return ext

    if mask.dim() == 2:
        # Simple padding mask — just prepend ones for the prefix positions.
        B, S = mask.shape
        pfx = torch.ones(B, n, dtype=mask.dtype, device=device)
        return torch.cat([pfx, mask], dim=1)

    raise ValueError(f"Unexpected mask shape: {mask.shape}")