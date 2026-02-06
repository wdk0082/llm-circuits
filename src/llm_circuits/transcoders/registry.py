"""Transcoder registry: maps model sizes to HF model ids and transcoder repos."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Describes a model + its matching transcoder repo."""

    size: str
    hf_model_id: str
    transcoder_repo: str


_REGISTRY: dict[str, ModelSpec] = {
    "0.6b": ModelSpec(
        size="0.6b",
        hf_model_id="Qwen/Qwen3-0.6B",
        transcoder_repo="mwhanna/qwen3-0.6b-transcoders-lowl0",
    ),
    "1.7b": ModelSpec(
        size="1.7b",
        hf_model_id="Qwen/Qwen3-1.7B",
        transcoder_repo="mwhanna/qwen3-1.7b-transcoders-lowl0",
    ),
    "4b": ModelSpec(
        size="4b",
        hf_model_id="Qwen/Qwen3-4B",
        transcoder_repo="mwhanna/qwen3-4b-transcoders",
    ),
    "8b": ModelSpec(
        size="8b",
        hf_model_id="Qwen/Qwen3-8B",
        transcoder_repo="mwhanna/qwen3-8b-transcoders",
    ),
    "14b": ModelSpec(
        size="14b",
        hf_model_id="Qwen/Qwen3-14B",
        transcoder_repo="mwhanna/qwen3-14b-transcoders-lowl0",
    ),
}


def get_spec(size: str) -> ModelSpec:
    """Look up a ``ModelSpec`` by normalised size key (e.g. ``"0.6b"``).

    Raises ``KeyError`` with a helpful message if the size is unknown.
    """
    key = size.lower().strip()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown model size {size!r}. Available: {available}")
    return _REGISTRY[key]


def list_specs() -> list[ModelSpec]:
    """Return all registered ``ModelSpec`` entries."""
    return list(_REGISTRY.values())
