"""Reproducibility helpers."""

from __future__ import annotations

import random

import torch


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for Python, PyTorch (CPU + CUDA)."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
