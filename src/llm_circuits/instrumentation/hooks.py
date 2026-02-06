"""Generic PyTorch forward-hook helpers.

These utilities are intentionally minimal and framework-agnostic --
they do not depend on TransformerLens, NNsight, or circuit-tracer.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch
from torch import Tensor, nn


@contextmanager
def attach_hook(
    module: nn.Module,
    fn: Callable[[nn.Module, tuple[Any, ...], Any], Any | None],
):
    """Context manager that registers a forward hook and removes it on exit.

    Args:
        module: The PyTorch module to hook.
        fn: A callable with signature ``(module, input, output) -> output | None``.

    Yields:
        The :class:`torch.utils.hooks.RemovableHook` handle.
    """
    handle = module.register_forward_hook(fn)
    try:
        yield handle
    finally:
        handle.remove()


class ActivationRecorder:
    """Record activations from named modules during a forward pass.

    Example::

        recorder = ActivationRecorder()
        with recorder.attach(model, ["model.layers.0.mlp", "model.layers.1.mlp"]):
            model(**inputs)
        print(recorder.activations.keys())
    """

    def __init__(self) -> None:
        self.activations: OrderedDict[str, Tensor] = OrderedDict()
        self._handles: list[torch.utils.hooks.RemovableHook] = []

    def _make_hook(self, name: str) -> Callable:
        def hook(_module: nn.Module, _input: tuple, output: Any) -> None:
            if isinstance(output, Tensor):
                self.activations[name] = output.detach()
            elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], Tensor):
                self.activations[name] = output[0].detach()

        return hook

    @contextmanager
    def attach(self, model: nn.Module, module_names: list[str]):
        """Context manager that hooks the requested modules and records outputs.

        Args:
            model: The root module (e.g. the full ``AutoModelForCausalLM``).
            module_names: Dot-separated module names to hook
                (e.g. ``"model.layers.0.mlp"``).
        """
        self.activations.clear()
        for name in module_names:
            submodule = model.get_submodule(name)
            handle = submodule.register_forward_hook(self._make_hook(name))
            self._handles.append(handle)
        try:
            yield self
        finally:
            for h in self._handles:
                h.remove()
            self._handles.clear()
