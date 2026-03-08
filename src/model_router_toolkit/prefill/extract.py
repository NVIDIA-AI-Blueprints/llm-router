"""Prefill feature extraction: run a single forward pass through an encoder model.

Loads a HuggingFace causal LM, runs the question through it with
output_hidden_states=True, and returns per-layer hidden states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class PrefillResult:
    hidden_last: dict[int, np.ndarray]
    hidden_mean: dict[int, np.ndarray]
    n_layers: int
    hidden_dim: int


class PrefillExtractor:
    """Loads an HF causal LM and extracts prefill hidden states."""

    def __init__(
        self,
        hf_path: str,
        *,
        device: str | None = None,
        dtype: Any = None,
        cache_dir: str | None = None,
    ):
        self._hf_path = hf_path
        self._cache_dir = cache_dir
        self._model = None
        self._tokenizer = None
        self.n_layers = 0
        self.hidden_dim = 0

        if device is None:
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"
        else:
            self._device = device

        self._dtype = dtype or (torch.float32 if self._device == "cpu" else torch.bfloat16)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._hf_path, cache_dir=self._cache_dir, trust_remote_code=True,
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            self._hf_path,
            dtype=self._dtype,
            device_map=self._device,
            cache_dir=self._cache_dir,
            trust_remote_code=True,
        )
        self._model.eval()

        cfg = self._model.config
        self.n_layers = cfg.num_hidden_layers
        self.hidden_dim = cfg.hidden_size

    def extract(
        self,
        question: str,
        *,
        chat_template_kwargs: dict | None = None,
        extract_layers: list[int] | None = None,
    ) -> PrefillResult:
        """Run prefill on a single question and return hidden states."""
        self._ensure_loaded()

        tpl_kwargs = chat_template_kwargs or {}
        formatted = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
            **tpl_kwargs,
        )

        inputs = self._tokenizer(
            formatted, return_tensors="pt", truncation=True, max_length=2048,
        )
        input_ids = inputs["input_ids"].to(self._model.device)
        attention_mask = inputs["attention_mask"].to(self._model.device)
        seq_len = int(attention_mask.sum())

        with torch.no_grad():
            outputs = self._model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        all_hidden = outputs.hidden_states  # tuple of (1, seq_len, hidden_dim)

        if extract_layers is None:
            extract_layers = list(range(len(all_hidden)))

        hidden_last: dict[int, np.ndarray] = {}
        hidden_mean: dict[int, np.ndarray] = {}

        for li in extract_layers:
            if li >= len(all_hidden):
                continue
            hs = all_hidden[li][0, :seq_len, :].float()
            hidden_last[li] = hs[-1].cpu().numpy().reshape(1, -1)
            hidden_mean[li] = hs.mean(dim=0).cpu().numpy().reshape(1, -1)

        return PrefillResult(
            hidden_last=hidden_last,
            hidden_mean=hidden_mean,
            n_layers=self.n_layers,
            hidden_dim=self.hidden_dim,
        )

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
