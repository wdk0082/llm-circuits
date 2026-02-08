"""Convenience wrappers for loading Qwen3 models via the transcoder registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llm_circuits.models.hf_loader import load_model_and_tokenizer
from llm_circuits.transcoders.registry import get_spec

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


def load_qwen3(
    size: str = "0.6b",
    *,
    dtype_str: str = "bf16",
    device_map: str = "auto",
    cache_dir: str | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a Qwen3 model by size key (e.g. ``"0.6b"``, ``"1.7b"``).

    Resolves the HF model id from the transcoder registry so that the model
    id stays consistent with the available transcoders.
    """
    spec = get_spec(size)
    return load_model_and_tokenizer(
        spec.hf_model_id, dtype_str=dtype_str, device_map=device_map, cache_dir=cache_dir
    )
