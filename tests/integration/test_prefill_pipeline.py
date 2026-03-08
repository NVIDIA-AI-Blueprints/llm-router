"""Integration tests for the full prefill pipeline: extraction, scoring, routing.

Tests marked 'slow' require the Qwen/Qwen3.5-0.8B encoder model (~1.6GB)
and a smoke checkpoint. Run with: pytest --run-slow
"""

import time

import numpy as np
import pytest
import torch

from model_router_toolkit.prefill.extract import (
    PrefillExtractor,
    PrefillResult,
    detect_device,
    normalize_question,
    prefill_cache_path,
    run_extraction,
    template_hash,
)

pytestmark = [pytest.mark.requires_torch]


class TestExtractUtilities:
    """Pure-function tests that don't need the encoder."""

    def test_normalize_question(self):
        assert normalize_question("  What  is  2+2? ") == "what is 2+2?"

    def test_template_hash_deterministic(self):
        h1 = template_hash("Qwen/Qwen3.5-0.8B", {"enable_thinking": True})
        h2 = template_hash("Qwen/Qwen3.5-0.8B", {"enable_thinking": True})
        assert h1 == h2
        assert len(h1) == 12

    def test_prefill_cache_path_format(self):
        p = prefill_cache_path("/tmp/cache", "Qwen/Qwen3.5-0.8B", {})
        assert p.parent.name == "cache"
        assert p.name.startswith("prefill_Qwen_Qwen3.5-0.8B_")
        assert p.suffix == ".pt"

    def test_detect_device(self):
        device = detect_device()
        assert device in ("cuda", "mps", "cpu")


class TestPrefillResultSerialization:
    def test_save_load_roundtrip(self, tmp_path):
        layers = [10, 11, 12]
        hidden_last = {li: torch.randn(5, 64) for li in layers}
        hidden_mean = {li: torch.randn(5, 64) for li in layers}
        original = PrefillResult(
            hidden_last=hidden_last, hidden_mean=hidden_mean,
            n_layers=24, hidden_dim=64,
        )

        save_path = tmp_path / "test_prefill.pt"
        original.save(save_path)
        loaded = PrefillResult.load(save_path)

        assert loaded.n_layers == 24
        assert loaded.hidden_dim == 64
        assert loaded.available_layers == layers
        for li in layers:
            torch.testing.assert_close(original.hidden_last[li], loaded.hidden_last[li])
            torch.testing.assert_close(original.hidden_mean[li], loaded.hidden_mean[li])


@pytest.mark.slow
class TestExtractorWithRealEncoder:
    """Tests requiring the actual Qwen/Qwen3.5-0.8B encoder model."""

    def test_extractor_batch_real_encoder(self):
        extractor = PrefillExtractor("Qwen/Qwen3.5-0.8B", device="cpu")
        questions = ["What is 2+2?", "Prove P=NP"]
        result = extractor.extract_batch(
            questions, batch_size=2, show_progress=False,
        )
        assert isinstance(result, PrefillResult)
        assert result.hidden_dim > 0
        assert result.n_layers > 0
        for li in result.available_layers:
            assert result.hidden_last[li].shape[0] == 2
            assert result.hidden_last[li].shape[1] == result.hidden_dim
        extractor.unload()

    def test_run_extraction_with_cache(self, tmp_path):
        questions = ["What is the capital of France?"]
        cache_dir = tmp_path / "prefill_cache"

        t0 = time.time()
        result1 = run_extraction(
            "Qwen/Qwen3.5-0.8B", questions,
            device="cpu", cache_dir=str(cache_dir),
        )
        first_time = time.time() - t0

        t0 = time.time()
        result2 = run_extraction(
            "Qwen/Qwen3.5-0.8B", questions,
            device="cpu", cache_dir=str(cache_dir),
        )
        cached_time = time.time() - t0

        assert cached_time < first_time
        for li in result1.available_layers:
            torch.testing.assert_close(result1.hidden_last[li], result2.hidden_last[li])


@pytest.mark.slow
class TestScorerWithRealCheckpoint:
    """Tests requiring encoder + smoke checkpoint."""

    def test_scorer_score_real_checkpoint(self, smoke_ckpt_path):
        from model_router_toolkit.prefill.scorer import PrefillScorer

        scorer = PrefillScorer(smoke_ckpt_path)
        scores = scorer.score("What is the capital of France?")
        assert len(scores.model_names) > 0
        assert len(scores.confidences) == len(scores.model_names)
        for c in scores.confidences:
            assert 0.0 <= c <= 1.0
        assert len(scores.costs) == len(scores.model_names)
        scorer.unload()

    def test_prefill_router_route_real(self, smoke_ckpt_path):
        from model_router_toolkit.prefill.router import PrefillRouter

        router = PrefillRouter()
        router.load(smoke_ckpt_path)
        result = router.route("Explain quantum entanglement", tolerance=0.20)
        assert result.selected_model in result.model_names
        assert len(result.confidences) == len(result.model_names)
        for c in result.confidences:
            assert 0.0 <= c <= 1.0
        assert "p_max" in result.metadata
        router.unload()

    def test_prefill_router_unload_frees_memory(self, smoke_ckpt_path):
        from model_router_toolkit.prefill.router import PrefillRouter

        router = PrefillRouter()
        router.load(smoke_ckpt_path)
        router.route("test", tolerance=0.5)
        router.unload()
        assert router._scorer is None
