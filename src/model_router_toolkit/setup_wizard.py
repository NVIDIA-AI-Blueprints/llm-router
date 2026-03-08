"""Interactive setup wizard for model-router-toolkit.

Detects environment, loads checkpoint to discover model names,
maps them to API endpoints, and generates pool_config.yaml.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import yaml

from model_router_toolkit.gpu import detect_gpus, has_sufficient_gpu

KNOWN_ENDPOINTS = {
    "nvidia": {
        "nem-think":   "nvidia/nemotron-3-nano-30b-a3b",
        "nem-nothink": "nvidia/nemotron-3-nano-30b-a3b",
        "nem-super":   "nvidia/llama-3.3-nemotron-super-49b-v1.5",
        "gptoss-high": "openai/gpt-oss-20b",
        "gptoss-20b":  "openai/gpt-oss-20b",
        "gptoss-120b": "openai/gpt-oss-120b",
        "qwen-122b":   "qwen/qwen3.5-122b-a10b",
        "gpt-5.2":     "openai/gpt-5.2",
        "claude-opus":  "aws/anthropic/bedrock-claude-opus-4-6",
    },
    "openrouter": {
        "nem-think":   "openrouter/nvidia/nemotron-3-nano-30b-a3b",
        "nem-nothink": "openrouter/nvidia/nemotron-3-nano-30b-a3b",
        "nem-super":   "openrouter/nvidia/llama-3.3-nemotron-super-49b-v1.5",
        "gptoss-high": "openrouter/openai/gpt-oss-20b",
        "gptoss-20b":  "openrouter/openai/gpt-oss-20b",
        "gptoss-120b": "openrouter/openai/gpt-oss-120b",
        "qwen-122b":   "openrouter/qwen/qwen3.5-122b-a10b",
        "gpt-5.2":     "openrouter/openai/gpt-5.2",
        "claude-opus":  "openrouter/anthropic/claude-opus-4.6",
    },
}

DISPLAY_NAMES = {
    "nem-think": "Nemotron 3 Nano Think",
    "nem-nothink": "Nemotron 3 Nano",
    "nem-super": "Nemotron 3 Super",
    "gptoss-high": "GPT-OSS 20B",
    "gptoss-20b": "GPT-OSS 20B",
    "gptoss-120b": "GPT-OSS 120B",
    "qwen-122b": "Qwen 3.5 122B",
    "gpt-5.2": "GPT-5.2",
    "claude-opus": "Claude Opus 4.6",
}

COSTS: dict[str, tuple[float, float]] = {
    "nem-think": (0.20, 0.20),
    "nem-nothink": (0.04, 0.16),
    "nem-super": (0.30, 0.30),
    "gptoss-high": (0.30, 0.30),
    "gptoss-20b": (0.30, 0.30),
    "gptoss-120b": (1.00, 1.00),
    "qwen-122b": (0.50, 0.50),
    "gpt-5.2": (1.75, 14.00),
    "claude-opus": (5.00, 25.00),
}

CHAT_TEMPLATE_KWARGS: dict[str, dict] = {
    "nem-think": {"enable_thinking": True},
    "gptoss-high": {"reasoning_effort": "high"},
    "gptoss-20b": {"reasoning_effort": "high"},
}

SYSTEM_PROMPTS: dict[str, str] = {
    "nem-think": "Think step-by-step before answering.",
    "nem-nothink": "Answer directly and concisely.",
}


def _read_models_from_pkl(path: Path) -> list[str]:
    with open(path, "rb") as f:
        db = pickle.load(f)
    return db["models"]


def _read_models_from_pt(path: Path) -> list[str]:
    try:
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except ImportError:
        print("  torch not installed -- enter model names manually.")
        return []
    return ckpt.get("model_names", [])


def _discover_checkpoint_models(checkpoint_path: Path) -> list[str]:
    suffix = checkpoint_path.suffix.lower()
    if suffix == ".pkl":
        return _read_models_from_pkl(checkpoint_path)
    elif suffix == ".pt":
        return _read_models_from_pt(checkpoint_path)
    else:
        print(f"  Unknown checkpoint format: {suffix}")
        return []


def _prompt_model_mapping(
    model_names: list[str],
    provider: str,
    api_base: str,
) -> list[dict[str, Any]]:
    """For each model in the checkpoint, ask the user to confirm or provide the API endpoint."""
    endpoint_map = KNOWN_ENDPOINTS.get(provider, {})
    models_yaml: list[dict[str, Any]] = []

    print(f"\nModel mapping ({len(model_names)} models from checkpoint):")
    print(f"  Provider: {provider} ({api_base})\n")

    for name in model_names:
        default_endpoint = endpoint_map.get(name, "")
        display = DISPLAY_NAMES.get(name, name)
        ci, co = COSTS.get(name, (0.0, 0.0))

        if default_endpoint:
            print(f"  {name} -> {default_endpoint}")
            confirm = input(f"    Accept? [Y/n/custom endpoint]: ").strip()
            if confirm.lower() == "n":
                continue
            elif confirm and confirm.lower() != "y":
                default_endpoint = confirm
        else:
            print(f"  {name} -- no known endpoint for provider '{provider}'")
            default_endpoint = input(f"    Enter litellm model string (or skip): ").strip()
            if not default_endpoint:
                print(f"    Skipping {name}")
                continue

        entry: dict[str, Any] = {
            "name": name,
            "display_name": display,
            "litellm_model": default_endpoint,
            "cost_per_m_input_tokens": ci,
            "cost_per_m_output_tokens": co,
        }
        if name in SYSTEM_PROMPTS:
            entry["system_prompt"] = SYSTEM_PROMPTS[name]
        if name in CHAT_TEMPLATE_KWARGS:
            entry["chat_template_kwargs"] = CHAT_TEMPLATE_KWARGS[name]

        models_yaml.append(entry)

    return models_yaml


def run_setup() -> None:
    """Run interactive setup and write configs/generated.yaml."""
    print("=" * 60)
    print("Model Router Toolkit -- Setup")
    print("=" * 60)

    # --- GPU detection ---
    print("\n1. GPU Detection")
    gpus = detect_gpus()
    if not gpus:
        print("  No NVIDIA GPUs found.")
    else:
        for g in gpus:
            print(f"  GPU {g.index}: {g.name} ({g.vram_gb:.1f} GB)")

    sufficient, _ = has_sufficient_gpu(min_vram_gb=16.0)

    # --- Routing method ---
    print("\n2. Routing Method")
    method = "kmeans"
    encoder = ""
    encoder_server = ""

    if sufficient:
        default_method = "prefill"
        print("  GPU detected -- prefill routing recommended (SOTA accuracy).")
    else:
        default_method = "kmeans"
        print("  No GPU >= 16GB detected.")
        print("  - KMeans: lightweight, no GPU needed, uses embedding API")
        print("  - Prefill: higher accuracy, runs encoder model locally (small models work on CPU)")

    choice = input(f"  Select [prefill/kmeans] (default: {default_method}): ").strip() or default_method
    method = "prefill" if choice.lower().startswith("p") else "kmeans"

    if method == "prefill":
        default_encoder = "Qwen/Qwen3.5-0.8B" if not sufficient else "Qwen/Qwen3.5-35B-A3B"
        encoder = input(f"  Encoder model HF path (default: {default_encoder}): ").strip() or default_encoder
        if not sufficient:
            print(f"  Note: {encoder} will run on CPU. First call may take ~30s to load.")
        encoder_server = ""

    # --- Checkpoint ---
    print("\n3. Checkpoint")
    if method == "kmeans":
        default_ckpt = "checkpoints/kmeans_c100_db.pkl"
        ckpt_path = input(f"  Path to .pkl checkpoint (default: {default_ckpt}): ").strip() or default_ckpt
    else:
        default_ckpt = ""
        ckpt_path = input("  Path to .pt checkpoint: ").strip()

    ckpt_path_obj = Path(ckpt_path)
    if ckpt_path_obj.exists():
        print(f"  Loading {ckpt_path_obj}...")
        checkpoint_models = _discover_checkpoint_models(ckpt_path_obj)
        if checkpoint_models:
            print(f"  Found {len(checkpoint_models)} models: {checkpoint_models}")
        else:
            print("  Could not read models from checkpoint.")
            raw = input("  Enter model names comma-separated: ").strip()
            checkpoint_models = [m.strip() for m in raw.split(",") if m.strip()]
    else:
        print(f"  Checkpoint not found at {ckpt_path}.")
        if method == "kmeans" and ckpt_path == default_ckpt:
            print("  Will use default checkpoint path in config (resolve at serve time).")
        checkpoint_models = []
        raw = input("  Enter model names comma-separated (or leave empty): ").strip()
        if raw:
            checkpoint_models = [m.strip() for m in raw.split(",") if m.strip()]

    # --- Provider ---
    print("\n4. Inference Provider")
    provider_choice = input("  Select [openrouter/nvidia] (default: openrouter): ").strip() or "openrouter"
    provider = "nvidia" if provider_choice.lower().startswith("n") else "openrouter"

    if provider == "openrouter":
        api_base = "https://openrouter.ai/api/v1"
        key_name = "OPENROUTER_API_KEY"
    else:
        api_base = "https://integrate.api.nvidia.com/v1"
        key_name = "NVIDIA_API_KEY"

    key = os.environ.get(key_name, "")
    if key:
        print(f"  {key_name} found in environment.")
    else:
        print(f"  {key_name} not set. Set it before running serve.")

    # --- Model mapping ---
    print("\n5. Model Mapping")
    if checkpoint_models:
        models_yaml = _prompt_model_mapping(checkpoint_models, provider, api_base)
    else:
        print("  No models from checkpoint. Using full default pool.")
        endpoint_map = KNOWN_ENDPOINTS.get(provider, {})
        default_pool = ["nem-think", "nem-super", "gptoss-20b", "gptoss-120b", "qwen-122b", "gpt-5.2", "claude-opus"]
        models_yaml = []
        for name in default_pool:
            ci, co = COSTS.get(name, (0.0, 0.0))
            entry: dict[str, Any] = {
                "name": name,
                "display_name": DISPLAY_NAMES.get(name, name),
                "litellm_model": endpoint_map.get(name, name),
                "cost_per_m_input_tokens": ci,
                "cost_per_m_output_tokens": co,
            }
            if name in SYSTEM_PROMPTS:
                entry["system_prompt"] = SYSTEM_PROMPTS[name]
            if name in CHAT_TEMPLATE_KWARGS:
                entry["chat_template_kwargs"] = CHAT_TEMPLATE_KWARGS[name]
            models_yaml.append(entry)

    # --- Build config ---
    routing: dict[str, Any] = {
        "method": method,
        "checkpoint": ckpt_path,
        "tolerance": 0.20,
    }
    if method == "kmeans":
        routing["embed_model"] = "nvidia/llama-nemotron-embed-1b-v2"
        routing["embed_mode"] = "api"
        routing["embed_api_base"] = api_base
    else:
        routing["encoder"] = encoder
        routing["encoder_server"] = encoder_server
        routing["training_mode"] = "auto"
        routing["encoder_backend"] = "transformers"

    config = {"routing": routing, "models": models_yaml}

    out_path = Path("configs/generated.yaml")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print(f"Config written to {out_path}")
    print(f"  Method:     {method}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Provider:   {provider}")
    print(f"  Models:     {len(models_yaml)}")
    for m in models_yaml:
        print(f"    {m['name']:<15} -> {m['litellm_model']}")
    print(f"\nReady! Run: model-router serve --config {out_path}")
