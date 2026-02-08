"""Convenience wrappers for loading Gemma2 models via the transcoder registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llm_circuits.models.hf_loader import load_model_and_tokenizer
from llm_circuits.transcoders.registry import get_spec

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


def load_gemma2(
    key: str = "gemma2-2b",
    *,
    dtype_str: str = "bf16",
    device_map: str = "auto",
    cache_dir: str | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a Gemma2 model by registry key (e.g. ``"gemma2-2b"``).

    Resolves the HF model id from the transcoder registry so that the model
    id stays consistent with the available transcoders.
    """
    spec = get_spec(key)
    return load_model_and_tokenizer(
        spec.hf_model_id, dtype_str=dtype_str, device_map=device_map, cache_dir=cache_dir
    )
