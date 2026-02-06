"""Intervention primitives for causal tracing and ablation studies.

This module will house our own intervention implementations,
independent of circuit-tracer's intervention machinery.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor, nn


@contextmanager
def ablate_module(
    model: nn.Module,
    module_name: str,
    replacement: Callable[[Tensor], Tensor] | Tensor | None = None,
):
    """Context manager that replaces a module's output during the forward pass.

    Args:
        model: The root model.
        module_name: Dot-separated name of the submodule to ablate.
        replacement: Either a callable ``(original_output) -> new_output``,
            a constant ``Tensor`` to substitute, or ``None`` for zero-ablation.

    Yields:
        The hook handle.
    """
    submodule = model.get_submodule(module_name)

    def hook(_mod: nn.Module, _inp: tuple[Any, ...], output: Any) -> Any:
        first = output[0] if isinstance(output, tuple) else output

        if replacement is None:
            result = torch.zeros_like(first)
        elif callable(replacement):
            result = replacement(first)
        else:
            result = replacement

        if isinstance(output, tuple):
            return (result, *output[1:])
        return result

    handle = submodule.register_forward_hook(hook)
    try:
        yield handle
    finally:
        handle.remove()
