"""Global settings resolved from environment variables and defaults."""

from __future__ import annotations

import os
from pathlib import Path

import torch


def default_device() -> str:
    """Return the best available device string."""
    override = os.environ.get("LLM_CIRCUITS_DEVICE")
    if override:
        return override
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def default_dtype() -> torch.dtype:
    """Return bfloat16 on CUDA/CPU, float32 on MPS (bf16 unsupported on older MPS)."""
    dev = default_device()
    if dev == "mps":
        return torch.float32
    return torch.bfloat16


def project_root() -> Path:
    """Return the repo root (two levels up from this file in a src-layout)."""
    return Path(__file__).resolve().parents[2]


def cache_dir() -> Path:
    """Return the default local cache directory for transcoders."""
    return project_root() / ".cache"
