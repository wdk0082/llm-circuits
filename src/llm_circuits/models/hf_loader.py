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
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a causal-LM and its tokenizer from the Hugging Face Hub.

    Args:
        model_id: Hub repo id, e.g. ``"Qwen/Qwen3-0.6B"``.
        dtype_str: One of ``"bf16"``, ``"fp16"``, ``"fp32"``.
        device_map: Passed to ``AutoModelForCausalLM.from_pretrained``.
        trust_remote_code: Whether to trust remote code in the repo.

    Returns:
        A ``(model, tokenizer)`` tuple ready for generation.
    """
    dtype = _DTYPE_MAP.get(dtype_str)
    if dtype is None:
        raise ValueError(f"Unknown dtype_str {dtype_str!r}. Choose from {list(_DTYPE_MAP)}")

    log.info("Loading model [bold]%s[/bold] (dtype=%s, device_map=%s)", model_id, dtype, device_map)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    return model, tokenizer
