"""Llama with SASRec embedding injection (Phase 3). Supports base HF or local PEFT LoRA."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adapter import AdapterConfig, EmbeddingAdapter
from .data import IdMaps, pick_device
from .ranking_data import RankingExample

DEFAULT_LLM_PATH = "llama31-1b-movielens-full-final"


def _dtype_for_device(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def load_llm_and_tokenizer(
    model_path: Union[str, Path],
    device: torch.device,
    dtype: torch.dtype,
):
    """
    Load HuggingFace causal LM or a local PEFT LoRA folder (adapter_config.json + adapter_model.*).
    """
    path = Path(model_path)
    if path.is_dir() and (path / "adapter_config.json").exists():
        cfg = json.loads((path / "adapter_config.json").read_text())
        base_name = cfg.get("base_model_name_or_path", "unsloth/Llama-3.2-1B-Instruct")
        print(f"[load] PEFT adapter from {path}")
        print(f"[load] base model {base_name} (download on first run ~2–5 min)")
        try:
            tokenizer = AutoTokenizer.from_pretrained(path)
        except (ValueError, OSError):
            tokenizer = AutoTokenizer.from_pretrained(base_name)
        base = AutoModelForCausalLM.from_pretrained(base_name, dtype=dtype)
        from peft import PeftModel

        llm = PeftModel.from_pretrained(base, str(path))
        llm = llm.to(device)
        print("[load] LLM ready (PEFT)")
        return llm, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).to(device)
    return llm, tokenizer


class InjectedLlamaRanker(nn.Module):
    """Frozen Llama + trainable adapter; inject collaborative vectors after history prefix."""

    def __init__(
        self,
        model_name: str | Path = DEFAULT_LLM_PATH,
        checkpoint_dir: str | Path = "checkpoints",
        device: Optional[torch.device] = None,
        freeze_llm: bool = True,
        train_adapter: bool = True,
        load_embedding_adapter: bool = True,
        embedding_adapter_path: str | Path | None = None,
    ):
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.device = device or pick_device()
        self.dtype = _dtype_for_device(self.device)
        self.model_path = Path(model_name)

        self.llm, self.tokenizer = load_llm_and_tokenizer(
            self.model_path, self.device, self.dtype
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False
            self.llm.eval()
            self._maybe_merge_peft_for_inference()

        id_maps = IdMaps.from_json(self.checkpoint_dir / "id_maps.json")
        self.id_maps = id_maps
        item_emb = torch.load(
            self.checkpoint_dir / "item_embeddings.pt",
            map_location="cpu",
            weights_only=True,
        ).float()
        self.register_buffer("item_embeddings", item_emb, persistent=False)

        llm_hidden = self._resolve_hidden_size()
        self.adapter = EmbeddingAdapter(
            AdapterConfig(sasrec_dim=item_emb.shape[1], llm_dim=llm_hidden)
        )
        if load_embedding_adapter:
            emb_path = Path(embedding_adapter_path) if embedding_adapter_path else self.checkpoint_dir / "adapter.pt"
            adapter_ckpt = torch.load(emb_path, map_location="cpu", weights_only=False)
            cfg = adapter_ckpt.get("config", {})
            if cfg.get("llm_dim"):
                self.adapter = EmbeddingAdapter(
                    AdapterConfig(
                        sasrec_dim=cfg.get("sasrec_dim", item_emb.shape[1]),
                        llm_dim=cfg["llm_dim"],
                        hidden_dim=cfg.get("hidden_dim", 1024),
                    )
                )
            state = adapter_ckpt.get("model_state_dict") or adapter_ckpt.get("adapter_state_dict")
            if state:
                self.adapter.load_state_dict(state)
        self.adapter.to(self.device)

        if not train_adapter:
            for p in self.adapter.parameters():
                p.requires_grad = False

        self.embed_layer = self.llm.get_input_embeddings()

    def _maybe_merge_peft_for_inference(self) -> None:
        """Merge LoRA into base weights so generate() works with inputs_embeds + chat."""
        try:
            from peft import PeftModel
        except ImportError:
            return
        if isinstance(self.llm, PeftModel):
            print("[load] merging LoRA into base model for inference...", flush=True)
            self.llm = self.llm.merge_and_unload()
            self.llm.eval()
            print("[load] merge complete", flush=True)

    def _resolve_hidden_size(self) -> int:
        base = getattr(self.llm, "base_model", None)
        if base is not None and hasattr(base, "config"):
            return base.config.hidden_size
        return self.llm.config.hidden_size

    @property
    def hidden_size(self) -> int:
        return self._resolve_hidden_size()

    def history_vectors(self, item_indices: List[int]) -> torch.Tensor:
        """item_indices: internal 1..n_items -> (N, hidden) projected vectors."""
        if not item_indices:
            return torch.zeros(0, self.hidden_size, device=self.device, dtype=self.dtype)
        idx = torch.tensor(item_indices, dtype=torch.long) - 1
        e = self.item_embeddings[idx].to(self.device, dtype=torch.float32)
        z = self.adapter.project(e).to(dtype=self.dtype)
        return z

    def _embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_layer(token_ids).to(dtype=self.dtype)

    def _supports_chat_template(self) -> bool:
        return bool(getattr(self.tokenizer, "chat_template", None))

    def _chat_format_user(self, content: str, add_generation_prompt: bool = False) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    def _tokenize_text(self, text: str) -> torch.Tensor:
        return self.tokenizer(
            text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids[0]

    def _chat_prefix_suffix_parts(self, example: RankingExample) -> Tuple[str, str, str]:
        """Split chat-formatted user prompt into prefix / suffix / assistant header strings."""
        fmt_prefix = self._chat_format_user(example.prefix_text, add_generation_prompt=False)
        fmt_user = self._chat_format_user(example.prompt, add_generation_prompt=False)
        fmt_gen = self._chat_format_user(example.prompt, add_generation_prompt=True)

        if fmt_user.startswith(fmt_prefix):
            suffix_part = fmt_user[len(fmt_prefix) :]
        else:
            suffix_part = example.suffix_text

        if fmt_gen.startswith(fmt_user):
            asst_part = fmt_gen[len(fmt_user) :]
        else:
            asst_part = ""

        return fmt_prefix, suffix_part, asst_part

    def build_sequence(
        self,
        example: RankingExample,
        target_position: Optional[int] = None,
        use_injection: bool = True,
        use_chat_template: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (input_embeds, attention_mask, labels) with labels=-100 for non-target tokens.
        """
        if use_chat_template and self._supports_chat_template():
            fmt_prefix, suffix_part, asst_part = self._chat_prefix_suffix_parts(example)
            prefix_ids = self._tokenize_text(fmt_prefix)
            suffix_ids = self._tokenize_text(suffix_part + asst_part)
        else:
            prefix_ids = self.tokenizer(
                example.prefix_text,
                add_special_tokens=True,
                return_tensors="pt",
            ).input_ids[0]
            suffix_ids = self.tokenizer(
                example.suffix_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids[0]

        prefix_emb = self._embed_tokens(prefix_ids.to(self.device))
        suffix_emb = self._embed_tokens(suffix_ids.to(self.device))

        parts = [prefix_emb]
        if use_injection:
            inject = self.history_vectors(example.history_item_indices)
            if inject.numel() > 0:
                # Single pooled profile vector (10 raw vectors often break generate)
                if inject.size(0) > 1:
                    inject = inject.mean(dim=0, keepdim=True)
                parts.append(inject)
        parts.append(suffix_emb)

        if target_position is not None:
            answer_text = f" {int(target_position)}"
            answer_ids = self.tokenizer(
                answer_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids[0].to(self.device)
            answer_emb = self._embed_tokens(answer_ids)
            parts.append(answer_emb)

        input_embeds = torch.cat(parts, dim=0)
        seq_len = input_embeds.size(0)
        attn = torch.ones(seq_len, dtype=torch.long, device=self.device)

        labels = torch.full((seq_len,), -100, dtype=torch.long, device=self.device)
        if target_position is not None:
            labels[-answer_ids.size(0) :] = answer_ids

        return input_embeds.unsqueeze(0), attn.unsqueeze(0), labels.unsqueeze(0)

    def forward_batch(
        self,
        examples: List[RankingExample],
        target_positions: Optional[List[int]] = None,
        use_injection: bool = True,
        rec_lambda: float = 0.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Pad-free batch size 1 style; returns (loss, rec_loss)."""
        assert len(examples) == 1, "use batch size 1 for variable-length injection"
        ex = examples[0]
        tgt = target_positions[0] if target_positions else None
        input_embeds, attn, labels = self.build_sequence(ex, tgt, use_injection=use_injection)

        outputs = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attn,
            labels=labels,
            return_dict=True,
        )
        loss = outputs.loss

        rec_loss = None
        if rec_lambda > 0 and ex.history_item_indices:
            idx = torch.tensor(ex.history_item_indices, dtype=torch.long) - 1
            e = self.item_embeddings[idx].to(self.device, dtype=torch.float32)
            rec_loss = self.adapter.reconstruction_loss(e)

        if rec_loss is not None and loss is not None:
            total = loss + rec_lambda * rec_loss
            return total, rec_loss
        return loss, rec_loss

    @torch.no_grad()
    def predict_position(
        self,
        example: RankingExample,
        use_injection: bool = True,
        max_new_tokens: int = 16,
        use_chat_template: bool = True,
        max_position: int = 10,
    ) -> Tuple[int, str]:
        """Returns (position 1-10, raw decoded text)."""
        self.llm.eval()
        if use_chat_template and self._supports_chat_template() and not use_injection:
            text = self._predict_text_chat(example, max_new_tokens)
            return parse_position_from_text(text, max_position), text

        input_embeds, attn, _ = self.build_sequence(
            example,
            target_position=None,
            use_injection=use_injection,
            use_chat_template=use_chat_template,
        )
        input_len = input_embeds.size(1)
        out = self.llm.generate(
            inputs_embeds=input_embeds,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = out[0, input_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        if use_injection and not text.strip():
            text = self._predict_text_chat(example, max_new_tokens)
            return parse_position_from_text(text, max_position), text
        return parse_position_from_text(text, max_position), text

    def _predict_text_chat(self, example: RankingExample, max_new_tokens: int) -> str:
        prompt_text = self._chat_format_user(example.prompt, add_generation_prompt=True)
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]
        out = self.llm.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        new_tokens = out[0, input_len:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


def parse_position_from_text(text: str, max_position: int = 10) -> int:
    """Extract first integer 1..max_position from model output; 0 means unparseable."""
    text = (text or "").strip()
    if not text:
        return 0
    opts = "|".join(str(i) for i in range(max_position, 0, -1))
    m = re.search(rf"\b({opts})\b", text)
    if m:
        return int(m.group(1))
    digits = re.findall(r"\d+", text)
    if digits:
        v = int(digits[0])
        if 1 <= v <= max_position:
            return v
    return 0
