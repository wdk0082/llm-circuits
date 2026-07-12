"""Tests for llm_circuits.utils (seeding / reproducibility)."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from llm_circuits.utils import seed_everything


def test_seed_everything_covers_python_numpy_torch():
    seed_everything(123)
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    seed_everything(123)
    b = (random.random(), np.random.rand(), torch.rand(1).item())
    assert a == b
    assert os.environ["PYTHONHASHSEED"] == "123"


def test_seed_everything_deterministic_flag_pins_cudnn():
    prev = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
    try:
        seed_everything(0, deterministic=True)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
    finally:
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = prev
