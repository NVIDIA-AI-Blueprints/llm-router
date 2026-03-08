"""Prefill feature extraction with batch support and caching.

Loads a HuggingFace causal LM, runs questions through it with
output_hidden_states=True, and returns per-layer hidden states.
Supports single-question extraction (inference via scorer) and
batch extraction (training/evaluation) with on-disk caching.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def normalize_question(q: str) -> str:
    """Normalize whitespace and case for question deduplication."""
    return " ".join(q.split()).strip().lower()


def template_hash(encoder: str, chat_template_kwargs: dict[str, Any]) -> str:
    """Short deterministic hash for an (encoder, template) combo."""
    key = f"{encoder}|{sorted(chat_template_kwargs.items())}"
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def prefill_cache_path(
    prefill_dir: str | Path,
    encoder: str,
    chat_template_kwargs: dict[str, Any],
) -> Path:
    """Cache filename keyed by (encoder, template)."""
    safe_enc = encoder.replace("/", "_").replace(" ", "_")
    th = template_hash(encoder, chat_template_kwargs)
    return Path(prefill_dir) / f"prefill_{safe_enc}_{th}.pt"


def detect_device() -> str:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# PrefillResult
# ---------------------------------------------------------------------------

@dataclass
class PrefillResult:
    """Raw hidden states for one set of questions through one encoder.

    Tensors are shape ``(N, hidden_dim)`` where N = number of questions.
    """

    hidden_last: dict[int, torch.Tensor]
    hidden_mean: dict[int, torch.Tensor]
    n_layers: int
    hidden_dim: int

    @property
    def available_layers(self) -> list[int]:
        return sorted(self.hidden_last.keys())

    def to_save_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "config": {"n_layers": self.n_layers, "hidden_dim": self.hidden_dim},
        }
        for li, t in self.hidden_last.items():
            data[f"layer_{li}"] = t
        for li, t in self.hidden_mean.items():
            data[f"layer_{li}_meanpool"] = t
        return data

    def save(self, path: str | os.PathLike) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.to_save_dict(), path)

    @classmethod
    def load(cls, path: str | os.PathLike) -> PrefillResult:
        data = torch.load(path, weights_only=False)
        return cls.load_from_dict(data)

    @classmethod
    def load_from_dict(cls, data: dict) -> PrefillResult:
        cfg = data.get("config", {})
        hidden_last: dict[int, torch.Tensor] = {}
        hidden_mean: dict[int, torch.Tensor] = {}
        for key in data:
            if key.startswith("layer_") and "_meanpool" not in key:
                li = int(key.split("_")[1])
                hidden_last[li] = data[key]
            elif key.endswith("_meanpool"):
                li = int(key.split("_")[1])
                hidden_mean[li] = data[key]
        n_layers = cfg.get(
            "n_layers", max(hidden_last.keys()) + 1 if hidden_last else 0,
        )
        sample = next(iter(hidden_last.values()))
        hidden_dim = cfg.get("hidden_dim", sample.shape[-1])
        return cls(
            hidden_last=hidden_last, hidden_mean=hidden_mean,
            n_layers=n_layers, hidden_dim=hidden_dim,
        )


# ---------------------------------------------------------------------------
# PrefillExtractor
# ---------------------------------------------------------------------------

class PrefillExtractor:
    """Loads an HF causal LM and extracts prefill hidden states.

    Supports both single-question extraction (for the scorer at inference)
    and batch extraction with progress bars (for training/eval).
    """

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

        self._device = device or detect_device()
        if dtype is not None:
            self._dtype = dtype
        elif self._device == "cpu":
            self._dtype = torch.float32
        else:
            self._dtype = torch.bfloat16

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        cd = self._cache_dir or os.environ.get("HF_HUB_CACHE")

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._hf_path, cache_dir=cd, trust_remote_code=True,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        load_kwargs: dict[str, Any] = {
            "dtype": self._dtype,
            "cache_dir": cd,
            "trust_remote_code": True,
        }
        if self._device != "cpu":
            load_kwargs["device_map"] = "auto"

        self._model = AutoModelForCausalLM.from_pretrained(
            self._hf_path, **load_kwargs,
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
        """Extract prefill features for a single question (scorer interface)."""
        return self.extract_batch(
            [question],
            chat_template_kwargs=chat_template_kwargs,
            extract_layers=extract_layers,
            batch_size=1,
            show_progress=False,
        )

    def extract_batch(
        self,
        questions: list[str],
        *,
        chat_template_kwargs: dict | None = None,
        extract_layers: list[int] | None = None,
        batch_size: int = 4,
        max_length: int = 2048,
        show_progress: bool = True,
    ) -> PrefillResult:
        """Extract prefill features for multiple questions with batching."""
        self._ensure_loaded()

        tpl_kwargs = chat_template_kwargs or {}
        if extract_layers is None:
            half = self.n_layers // 2
            extract_layers = list(range(half, self.n_layers))
        layers = extract_layers

        formatted = [
            self._tokenizer.apply_chat_template(
                [{"role": "user", "content": q}],
                tokenize=False, add_generation_prompt=True, **tpl_kwargs,
            )
            for q in questions
        ]

        all_last: dict[int, list[torch.Tensor]] = {li: [] for li in layers}
        all_mean: dict[int, list[torch.Tensor]] = {li: [] for li in layers}

        n_total = len(formatted)
        n_batches = (n_total + batch_size - 1) // batch_size
        iterator = range(0, n_total, batch_size)

        if show_progress:
            from tqdm import tqdm
            short_name = self._hf_path.split("/")[-1]
            iterator = tqdm(
                iterator, total=n_batches,
                desc=f"  extract({short_name})",
            )

        for batch_start in iterator:
            batch_texts = formatted[batch_start: batch_start + batch_size]
            inputs = self._tokenizer(
                batch_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=max_length,
            )
            input_ids = inputs["input_ids"].to(self._model.device)
            attention_mask = inputs["attention_mask"].to(self._model.device)

            with torch.no_grad():
                outputs = self._model(
                    input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True, use_cache=False,
                )

            hidden_states = outputs.hidden_states
            seq_lengths = attention_mask.sum(dim=1)

            for b in range(input_ids.shape[0]):
                seq_len = int(seq_lengths[b].item())
                for li in layers:
                    hs = hidden_states[li][b, :seq_len, :].float()
                    all_last[li].append(hs[-1].cpu())
                    all_mean[li].append(hs.mean(dim=0).cpu())

            del outputs, hidden_states, input_ids, attention_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return PrefillResult(
            hidden_last={li: torch.stack(all_last[li]) for li in layers},
            hidden_mean={li: torch.stack(all_mean[li]) for li in layers},
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


# ---------------------------------------------------------------------------
# High-level extraction with caching
# ---------------------------------------------------------------------------

def run_extraction(
    encoder_hf_path: str,
    questions: list[str],
    *,
    chat_template_kwargs: dict[str, Any] | None = None,
    device: str = "cpu",
    batch_size: int = 4,
    cache_dir: str | Path | None = None,
    hf_cache_dir: str | None = None,
) -> PrefillResult:
    """Extract prefill features for a list of questions, with caching."""
    tpl = chat_template_kwargs or {}

    if cache_dir:
        cp = prefill_cache_path(cache_dir, encoder_hf_path, tpl)
        if cp.exists():
            print(f"  Loading cached prefill: {cp}")
            return PrefillResult.load(cp)

    print(
        f"  Extracting prefill: {encoder_hf_path} "
        f"({len(questions)} questions, device={device})",
    )
    extractor = PrefillExtractor(
        encoder_hf_path, device=device, cache_dir=hf_cache_dir,
    )
    result = extractor.extract_batch(
        questions, chat_template_kwargs=tpl, batch_size=batch_size,
    )
    extractor.unload()

    if cache_dir:
        cp = prefill_cache_path(cache_dir, encoder_hf_path, tpl)
        result.save(cp)
        print(f"  Saved prefill cache: {cp}")

    return result


def extract_from_checkpoint(
    ckpt: dict[str, Any],
    questions: list[str],
    *,
    device: str = "cpu",
    batch_size: int = 4,
    cache_dir: str | Path | None = None,
    hf_cache_dir: str | None = None,
) -> dict[str, PrefillResult]:
    """Extract prefill features based on a trained checkpoint's transforms.

    Deduplicates by (encoder, template) so each encoder runs at most once.
    Returns ``{target_name: PrefillResult}``.
    """
    seen: dict[str, PrefillResult] = {}
    results: dict[str, PrefillResult] = {}

    for tname, t in ckpt.get("transforms", {}).items():
        enc = t.get("encoder", "")
        tpl = t.get("chat_template_kwargs", {})
        key = template_hash(enc, tpl)

        if key not in seen:
            seen[key] = run_extraction(
                enc, questions, chat_template_kwargs=tpl,
                device=device, batch_size=batch_size,
                cache_dir=cache_dir, hf_cache_dir=hf_cache_dir,
            )
        results[tname] = seen[key]

    return results
