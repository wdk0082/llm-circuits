"""Generic Hugging Face model + tokenizer loading via Transformers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_circuits.logging import get_logger

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

log = get_logger(__name__)

_DTYPE_MAP: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def load_model_and_tokenizer(
    model_id: str,
    *,
    dtype_str: str = "bf16",
    device_map: str = "auto",
    trust_remote_code: bool = True,
    cache_dir: str | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a causal-LM and its tokenizer from the Hugging Face Hub.

    Args:
        model_id: Hub repo id, e.g. ``"Qwen/Qwen3-0.6B"``.
        dtype_str: One of ``"bf16"``, ``"fp16"``, ``"fp32"``.
        device_map: Passed to ``AutoModelForCausalLM.from_pretrained``.
            ``"xla"``/``"tpu"`` is handled specially: accelerate's device_map
            machinery does not know XLA, so the model is loaded on CPU and
            moved to the XLA device afterwards.
        trust_remote_code: Whether to trust remote code in the repo.
        cache_dir: Local directory for cached model weights. Defaults to
            :func:`~llm_circuits.settings.model_cache_dir`.

    Returns:
        A ``(model, tokenizer)`` tuple ready for generation.
    """
    from llm_circuits.settings import model_cache_dir, xla_device

    dtype = _DTYPE_MAP.get(dtype_str)
    if dtype is None:
        raise ValueError(f"Unknown dtype_str {dtype_str!r}. Choose from {list(_DTYPE_MAP)}")

    xla_target = isinstance(device_map, str) and device_map.strip().lower() in ("xla", "tpu")
    if xla_target:
        xla_device()  # ensure torch_xla is imported before any .to("xla")
        device_map = None  # load on CPU, move to XLA below

    resolved_cache_dir = cache_dir if cache_dir is not None else str(model_cache_dir())

    log.info(
        "Loading model [bold]%s[/bold] (dtype=%s, device_map=%s, cache_dir=%s)",
        model_id,
        dtype,
        device_map,
        resolved_cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code, cache_dir=resolved_cache_dir
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,  # `torch_dtype` is deprecated since transformers 4.56 (see pyproject floor)
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        cache_dir=resolved_cache_dir,
    )
    if xla_target:
        model = model.to("xla")
    return model, tokenizer
