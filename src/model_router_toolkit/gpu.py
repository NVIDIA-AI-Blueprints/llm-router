"""GPU detection and VRAM checking utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GPUInfo:
    name: str
    vram_mb: int
    index: int

    @property
    def vram_gb(self) -> float:
        return self.vram_mb / 1024


def detect_gpus() -> list[GPUInfo]:
    """Detect available NVIDIA GPUs and their VRAM."""
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []

        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append(
                    GPUInfo(
                        index=int(parts[0]),
                        name=parts[1],
                        vram_mb=int(parts[2]),
                    )
                )
        return gpus
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []


def has_sufficient_gpu(min_vram_gb: float = 16.0) -> tuple[bool, list[GPUInfo]]:
    """Check if any GPU meets the minimum VRAM requirement."""
    gpus = detect_gpus()
    sufficient = [g for g in gpus if g.vram_gb >= min_vram_gb]
    return len(sufficient) > 0, gpus
