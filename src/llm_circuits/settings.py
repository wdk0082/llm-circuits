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
    """Return the base local cache directory.

    Priority:
      1. ``LLM_CIRCUITS_CACHE_DIR`` env var
      2. ``<project_root>/.cache``
    """
    override = os.environ.get("LLM_CIRCUITS_CACHE_DIR")
    if override:
        return Path(override)
    return project_root() / ".cache"


def artifacts_dir() -> Path:
    """Return the directory for output artifacts (plots, saved graphs, etc.).

    Priority:
      1. ``LLM_CIRCUITS_ARTIFACTS_DIR`` env var
      2. ``<project_root>/artifacts``
    """
    override = os.environ.get("LLM_CIRCUITS_ARTIFACTS_DIR")
    if override:
        return Path(override)
    return project_root() / "artifacts"


def transcoder_cache_dir() -> Path:
    """Return the default cache directory for transcoder weights."""
    return cache_dir() / "transcoders"


def model_cache_dir() -> Path:
    """Return the default cache directory for HF model weights."""
    return cache_dir() / "models"
