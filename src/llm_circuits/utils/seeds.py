"""Reproducibility helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, *, deterministic: bool = False) -> None:
    """Seed every RNG the toolkit touches: Python's ``random``, ``PYTHONHASHSEED``,
    NumPy, and PyTorch (CPU + all CUDA devices).

    ``deterministic=True`` additionally pins cuDNN to deterministic kernels
    (``cudnn.deterministic = True``, ``cudnn.benchmark = False``) — the GPU-rerun
    reproducibility knob the reproduction notebooks set at startup. It deliberately
    does NOT call ``torch.use_deterministic_algorithms``: that raises (or warns) on
    ops without deterministic implementations and requires ``CUBLAS_WORKSPACE_CONFIG``
    to be exported before CUDA initialization, neither of which suits notebook use;
    bf16 forwards on fixed hardware are bit-stable with the cuDNN pins alone
    (measured across reruns and A100 nodes — see DEVLOG).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
