"""Llama with SASRec embedding injection (Phase 3). Supports base HF or local PEFT LoRA.

Includes letter-based (A–J) log-prob ranking for Mode A (text) and Mode C (soft tokens).
"""

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

LETTERS = "ABCDEFGHIJ"


def render_mode_a_prompt(example: "RankingExample") -> str:
    """Build the raw Mode A text prompt (letter labels A-J, ends with 'Answer:').

    This is the single source of truth for the Mode A prompt format. Both the
    eval scorer (``InjectedLlamaRanker.build_mode_a_prompt``) and the LoRA
    ranking-finetune script (``scripts/finetune_lora_ranking.py``) call this so
    training and evaluation see byte-identical prompts (no train/eval drift).
    Chat-template wrapping, if any, is applied by the caller.
    """
    titles = _candidate_title_lines_from_suffix(example.suffix_text)
    n = len(titles)
    if n == 0:
        # Fallback: use movie IDs if we can't parse titles
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
        projected_embeddings_path: str | Path | None = None,
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

        # Pre-cached adapter projections (shape: n_items x llm_dim).
        # Created by scripts/cache_projected_embeddings.py — optional but
        # required for injection modes C and D (candidate soft tokens).
        proj_path = (
            Path(projected_embeddings_path)
            if projected_embeddings_path is not None
            else self.checkpoint_dir / "projected_embeddings.pt"
        )
        if proj_path.exists():
            proj_emb = torch.load(proj_path, map_location="cpu", weights_only=True)
            self.register_buffer("projected_embeddings", proj_emb, persistent=False)
            print(f"[load] projected_embeddings cache loaded {tuple(proj_emb.shape)}", flush=True)
        else:
            self.projected_embeddings = None

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
        self.train_adapter = train_adapter

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
        """item_indices: internal 1..n_items -> (N, hidden) projected vectors.
        Uses projected_embeddings cache when available; falls back to live adapter."""
        if not item_indices:
            return torch.zeros(0, self.hidden_size, device=self.device, dtype=self.dtype)
        idx = torch.tensor(item_indices, dtype=torch.long) - 1
        if self.projected_embeddings is not None:
            return self.projected_embeddings[idx].to(self.device, dtype=self.dtype)
        e = self.item_embeddings[idx].to(self.device, dtype=torch.float32)
        return self.adapter.project(e).to(dtype=self.dtype)

    def candidate_vectors(self, candidate_movie_ids: List[int]) -> List[torch.Tensor]:
        """Map candidate MovieIDs -> list of (1, llm_dim) projected tensors.
        Uses pre-cached projected_embeddings (must exist for C/D modes) unless
        train_adapter is True, in which case it dynamically projects from SASRec."""
        if self.projected_embeddings is None and not self.train_adapter:
            raise RuntimeError(
                "projected_embeddings.pt not found. "
                "Run: python scripts/cache_projected_embeddings.py"
            )
        vecs = []
        for mid in candidate_movie_ids:
            idx = self.id_maps.movie_to_idx.get(int(mid))
            if idx is None:
                # Unknown movie: use zero vector
                vecs.append(torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype))
            else:
                if self.projected_embeddings is not None and not self.train_adapter:
                    vecs.append(
                        self.projected_embeddings[idx - 1]
                        .unsqueeze(0)
                        .to(self.device, dtype=self.dtype)
                    )
                else:
                    e = self.item_embeddings[idx - 1].to(self.device, dtype=torch.float32)
                    v = self.adapter.project(e)
                    vecs.append(v.unsqueeze(0).to(dtype=self.dtype))
        return vecs

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

    def _build_candidate_suffix_with_soft_tokens(
        self,
        example: RankingExample,
        asst_part: str = "",
    ) -> torch.Tensor:
        """Build the candidate suffix embedding where each numbered line is
        replaced by its projected soft token.  Layout per candidate i:
            embed("\n{i}. ") | soft_token_i
        Followed by the closing instruction text.
        """
        cand_vecs = self.candidate_vectors(example.candidate_movie_ids)
        parts = []
        intro_ids = self._tokenize_text(
            "\n\nFrom the list below, rank which movie this user would most likely enjoy:"
        )
        parts.append(self._embed_tokens(intro_ids.to(self.device)))
        n = len(example.candidate_movie_ids)
        for i, vec in enumerate(cand_vecs, start=1):
            bullet_ids = self._tokenize_text(f"\n{i}. ")
            parts.append(self._embed_tokens(bullet_ids.to(self.device)))
            parts.append(vec)  # (1, llm_dim) soft token
        footer_ids = self._tokenize_text(
            f"\n\nReply with just the number (1-{n}) of the movie they would rate highest."
            + asst_part
        )
        parts.append(self._embed_tokens(footer_ids.to(self.device)))
        return torch.cat(parts, dim=0)  # (T_suffix, llm_dim)

    def build_sequence(
        self,
        example: RankingExample,
        target_position: Optional[int] = None,
        use_injection: bool = True,
        injection_mode: str = "history",
        use_chat_template: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (input_embeds, attention_mask, labels).

        injection_mode:
            'history'    (Mode B) — mean-pooled history soft token inserted between
                          prefix and suffix. Candidates remain as text.
            'candidates' (Mode C) — history stays as text. Each candidate line
                          is replaced by its projected soft token.
            'both'       (Mode D) — both history and candidates are soft tokens.
        """
        use_cand_tokens = use_injection and injection_mode in ("candidates", "both")
        use_hist_tokens = use_injection and injection_mode in ("history", "both")

        asst_part = ""
        if use_chat_template and self._supports_chat_template():
            fmt_prefix, suffix_part, asst_part = self._chat_prefix_suffix_parts(example)
            prefix_ids = self._tokenize_text(fmt_prefix)
        else:
            fmt_prefix = example.prefix_text
            suffix_part = example.suffix_text
            prefix_ids = self.tokenizer(
                fmt_prefix,
                add_special_tokens=True,
                return_tensors="pt",
            ).input_ids[0]

        prefix_emb = self._embed_tokens(prefix_ids.to(self.device))

        parts = [prefix_emb]

        # ── History injection (Mode B / D) ────────────────────────────
        if use_hist_tokens and example.history_item_indices:
            inject = self.history_vectors(example.history_item_indices)
            if inject.numel() > 0:
                parts.append(inject)

        # ── Candidate suffix ──────────────────────────────────────────
        if use_cand_tokens:
            # Mode C / D: replace candidate text lines with soft tokens
            cand_emb = self._build_candidate_suffix_with_soft_tokens(example, asst_part)
            parts.append(cand_emb)
        else:
            # Mode A (text-only) / B (history tokens, text candidates)
            suffix_ids = self._tokenize_text(suffix_part + asst_part)
            parts.append(self._embed_tokens(suffix_ids.to(self.device)))

        # ── Optional teacher-forced answer token (training only) ──────
        if target_position is not None:
            answer_text = f" {int(target_position)}"
            answer_ids = self.tokenizer(
                answer_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids[0].to(self.device)
            answer_emb = self._embed_tokens(answer_ids)
            parts.append(answer_emb)
            # Rebuild with answer appended
            input_embeds = torch.cat(parts, dim=0)
            seq_len = input_embeds.size(0)
            attn = torch.ones(seq_len, dtype=torch.long, device=self.device)
            labels = torch.full((seq_len,), -100, dtype=torch.long, device=self.device)
            labels[-answer_ids.size(0):] = answer_ids
            return input_embeds.unsqueeze(0), attn.unsqueeze(0), labels.unsqueeze(0)

        # ── Inference path: no target token appended ──────────────────
        input_embeds = torch.cat(parts, dim=0)
        seq_len = input_embeds.size(0)
        attn = torch.ones(seq_len, dtype=torch.long, device=self.device)
        labels = torch.full((seq_len,), -100, dtype=torch.long, device=self.device)
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
        injection_mode: str = "history",
        max_new_tokens: int = 16,
        use_chat_template: bool = True,
        max_position: int = 10,
    ) -> Tuple[int, str]:
        """Returns (position 1-10, raw decoded text).

        injection_mode: 'history' (B), 'candidates' (C), 'both' (D).
        """
        self.llm.eval()
        if use_chat_template and self._supports_chat_template() and not use_injection:
            text = self._predict_text_chat(example, max_new_tokens)
            return parse_position_from_text(text, max_position), text

        input_embeds, attn, _ = self.build_sequence(
            example,
            target_position=None,
            use_injection=use_injection,
            injection_mode=injection_mode,
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

    # ══════════════════════════════════════════════════════════════════
    #  Letter-based log-prob ranking (Mode A vs Mode C)
    # ══════════════════════════════════════════════════════════════════

    LETTERS = "ABCDEFGHIJ"

    def _get_letter_token_ids(self, n: int = 10) -> torch.Tensor:
        """Resolve the single token ID for each letter label ' A' .. ' J'.

        Returns a (n,) int64 tensor.  Cached after the first call.
        """
        cache_attr = "_letter_ids_cache"
        if hasattr(self, cache_attr):
            cached = getattr(self, cache_attr)
            if cached.size(0) >= n:
                return cached[:n].to(self.device)

        ids = []
        for ch in self.LETTERS[:n]:
            toks = self.tokenizer.encode(f" {ch}", add_special_tokens=False)
            if len(toks) != 1:
                raise RuntimeError(
                    f"Letter ' {ch}' tokenises to {toks} (expected single token). "
                    "Check tokenizer or switch to a different label scheme."
                )
            ids.append(toks[0])
        t = torch.tensor(ids, dtype=torch.long)
        setattr(self, cache_attr, t)
        return t.to(self.device)

    # ── Mode A: pure-text prompt with letter labels ──────────────────

    def build_mode_a_prompt(
        self, example: RankingExample, use_chat_template: bool = False,
    ) -> str:
        """Rebuild the prompt with A–J letter labels instead of 1–10 numbers.

        Returns the final string (chat-template-wrapped if use_chat_template=True
        and the tokenizer supports it; raw otherwise).
        """
        raw_prompt = render_mode_a_prompt(example)

        if use_chat_template and self._supports_chat_template():
            return self._chat_format_user(raw_prompt, add_generation_prompt=True)
        return raw_prompt

    # ── Mode C: soft-token candidates with letter labels ─────────────

    def build_mode_c_embeds(
        self, example: RankingExample, use_chat_template: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build inputs_embeds for Mode C (candidates as soft tokens).

        History stays as text.  Each candidate is a letter bullet + soft token.
        Returns (input_embeds, attention_mask) both with batch dim 1.
        """
        n = len(example.candidate_movie_ids)
        letter_range = f"{self.LETTERS[0]}-{self.LETTERS[n - 1]}"

        # ── Prefix: history as text ───────────────────────────────────
        if use_chat_template and self._supports_chat_template():
            # Split the chat template around the candidate section.
            placeholder = "<<CANDIDATES_PLACEHOLDER>>"
            raw_user = example.prefix_text + placeholder
            fmt_full = self._chat_format_user(raw_user, add_generation_prompt=True)
            before_ph, after_ph = fmt_full.split(placeholder, 1)
            prefix_ids = self._tokenize_text(before_ph)
            suffix_text_after = after_ph
        else:
            prefix_ids = self.tokenizer(
                example.prefix_text, add_special_tokens=True, return_tensors="pt"
            ).input_ids[0]
            suffix_text_after = ""

        parts: List[torch.Tensor] = [
            self._embed_tokens(prefix_ids.to(self.device))
        ]

        # ── Framing sentence ──────────────────────────────────────────
        framing = (
            "\n\nEach candidate below is represented as a collaborative filtering "
            "embedding that encodes viewing patterns from similar users:"
        )
        framing_ids = self._tokenize_text(framing)
        parts.append(self._embed_tokens(framing_ids.to(self.device)))

        # ── Candidate bullets: letter + soft token ────────────────────
        cand_vecs = self.candidate_vectors(example.candidate_movie_ids)
        for i, vec in enumerate(cand_vecs):
            bullet_ids = self._tokenize_text(f"\n{self.LETTERS[i]}. ")
            parts.append(self._embed_tokens(bullet_ids.to(self.device)))
            parts.append(vec)  # (1, llm_dim)

        # ── Footer ────────────────────────────────────────────────────
        footer = (
            f"\n\nReply with just the letter ({letter_range}) of the movie "
            "they would rate highest.\nAnswer:"
            + suffix_text_after
        )
        footer_ids = self._tokenize_text(footer)
        parts.append(self._embed_tokens(footer_ids.to(self.device)))

        input_embeds = torch.cat(parts, dim=0)  # (T, llm_dim)
        seq_len = input_embeds.size(0)
        attn = torch.ones(seq_len, dtype=torch.long, device=self.device)
        return input_embeds.unsqueeze(0), attn.unsqueeze(0)

    # ── Unified log-prob scoring ──────────────────────────────────────

    @torch.no_grad()
    def rank_by_logprob(
        self,
        example: RankingExample,
        mode: str = "text",
        use_chat_template: bool = False,
    ) -> Tuple[List[int], List[float]]:
        """Score candidates by log-prob of letter tokens A–J.

        Args:
            mode: 'text' (Mode A) or 'candidates' (Mode C).
            use_chat_template: wrap prompt in chat template if tokenizer supports it.
                Default False (raw prompts) — safer for pipeline validation.

        Returns:
            ranked_indices: candidate indices sorted by descending probability (0-based).
            probs: probability for each candidate position [P(A), P(B), ..., P(J)].
        """
        self.llm.eval()
        n = len(example.candidate_movie_ids)
        letter_ids = self._get_letter_token_ids(n)

        if mode == "text":
            prompt_text = self.build_mode_a_prompt(example, use_chat_template=use_chat_template)
            inputs = self.tokenizer(
                prompt_text, return_tensors="pt"
            ).to(self.device)
            outputs = self.llm(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                return_dict=True,
            )
        elif mode == "candidates":
            input_embeds, attn = self.build_mode_c_embeds(example, use_chat_template=use_chat_template)
            outputs = self.llm(
                inputs_embeds=input_embeds,
                attention_mask=attn,
                return_dict=True,
            )
        else:
            raise ValueError(f"Unknown mode '{mode}'; expected 'text' or 'candidates'")

        # Extract logits at the last token position
        logits = outputs.logits[0, -1, :]  # (vocab_size,)
        letter_logits = logits[letter_ids]  # (n,)
        probs = torch.softmax(letter_logits.float(), dim=0)

        ranked = torch.argsort(probs, descending=True).tolist()
        return ranked, probs.tolist()

    def forward_contrastive_loss(
        self,
        example: RankingExample,
        use_chat_template: bool = False,
        recon_lambda: float = 0.0,
        true_title: Optional[str] = None,
    ) -> torch.Tensor:
        """Listwise ranking CE over A-J letter logits, with optional reconstruction term.

        Args:
            recon_lambda: weight for reconstruction CE (title grounding). 0 = pure ranking.
            true_title: movie title string for the true candidate; required when recon_lambda > 0.
        """
        assert self.train_adapter, "train_adapter must be True to track gradients"

        # ── Listwise ranking loss ─────────────────────────────────────
        # build_mode_c_embeds calls candidate_vectors which uses live adapter
        # (because train_adapter=True bypasses the cache even if loaded).
        input_embeds, attn = self.build_mode_c_embeds(example, use_chat_template=use_chat_template)
        outputs = self.llm(
            inputs_embeds=input_embeds,
            attention_mask=attn,
            return_dict=True,
        )
        logits = outputs.logits[0, -1, :]  # (vocab_size,)
        n = len(example.candidate_movie_ids)
        letter_ids = self._get_letter_token_ids(n)
        letter_logits = logits[letter_ids]  # (n,)
        target_idx = example.true_position - 1
        rank_loss = torch.nn.functional.cross_entropy(
            letter_logits.unsqueeze(0).float(),
            torch.tensor([target_idx], dtype=torch.long, device=self.device),
        )

        if recon_lambda <= 0.0 or true_title is None:
            return rank_loss

        # ── Reconstruction grounding term ─────────────────────────────
        # Teacher-force the true candidate's title from its soft token.
        # Gradient flows only through adapter.project(e) -> z (soft token).
        true_mid = int(example.true_positive_movie_id)
        true_map_idx = self.id_maps.movie_to_idx.get(true_mid)
        if true_map_idx is None:
            return rank_loss  # OOV true item: skip recon, return just rank loss

        e = self.item_embeddings[true_map_idx - 1].unsqueeze(0).to(self.device, dtype=torch.float32)
        z = self.adapter.project(e).to(dtype=self.dtype)  # (1, llm_dim) — in-graph

        token_ids = self.tokenizer(
            true_title, add_special_tokens=False, return_tensors="pt"
        ).input_ids.to(self.device)
        with torch.no_grad():
            token_embs = self.embed_layer(token_ids).to(dtype=self.dtype)  # (1, T, D)

        inputs_embeds_recon = torch.cat([z.unsqueeze(1), token_embs], dim=1)  # (1, 1+T, D)
        ignore = torch.full((1, 1), -100, dtype=torch.long, device=self.device)
        labels = torch.cat([ignore, token_ids], dim=1)

        recon_out = self.llm(inputs_embeds=inputs_embeds_recon, labels=labels, return_dict=True)
        return rank_loss + recon_lambda * recon_out.loss


def _candidate_title_lines_from_suffix(suffix_text: str) -> List[str]:
    """Extract 'Title (genres)' from numbered lines in the candidate block."""
    titles: List[str] = []
    for line in suffix_text.split("\n"):
        line = line.strip()
        m = re.match(r"^\d+\.\s+(.+)$", line)
        if m:
            titles.append(m.group(1))
    return titles


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
