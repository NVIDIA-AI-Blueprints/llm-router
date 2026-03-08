"""Tests for GPU detection module."""

from unittest.mock import patch

from model_router_toolkit.gpu import GPUInfo, detect_gpus, has_sufficient_gpu


class TestGPU:
    def test_detect_gpus_returns_list(self):
        gpus = detect_gpus()
        assert isinstance(gpus, list)
        for g in gpus:
            assert isinstance(g, GPUInfo)

    def test_detect_gpus_no_nvidia_smi(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            gpus = detect_gpus()
        assert gpus == []

    def test_gpu_info_vram_gb(self):
        info = GPUInfo(name="Test GPU", vram_mb=16384, index=0)
        assert info.vram_gb == 16.0

    def test_has_sufficient_gpu_true(self):
        fake_gpu = GPUInfo(name="A100", vram_mb=24576, index=0)
        with patch("model_router_toolkit.gpu.detect_gpus", return_value=[fake_gpu]):
            sufficient, gpus = has_sufficient_gpu(min_vram_gb=16.0)
        assert sufficient is True
        assert len(gpus) == 1

    def test_has_sufficient_gpu_false(self):
        fake_gpu = GPUInfo(name="T4", vram_mb=4096, index=0)
        with patch("model_router_toolkit.gpu.detect_gpus", return_value=[fake_gpu]):
            sufficient, gpus = has_sufficient_gpu(min_vram_gb=16.0)
        assert sufficient is False
        assert len(gpus) == 1
