"""Transcoder registry: maps model keys to HF model ids and transcoder repos."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    """Describes a model + its matching transcoder repo."""

    size: str
    hf_model_id: str
    transcoder_repo: str
    family: str = "qwen3"
    transcoder_type: str = "per-layer"


# Canonical entries keyed by family-prefixed key.  Bare size aliases are added
# below for backward compatibility with existing Qwen3-only callers.
_REGISTRY: dict[str, ModelSpec] = {
    "qwen3-0.6b": ModelSpec(
        size="0.6b",
        hf_model_id="Qwen/Qwen3-0.6B",
        transcoder_repo="mwhanna/qwen3-0.6b-transcoders-lowl0",
        family="qwen3",
    ),
    "qwen3-1.7b": ModelSpec(
        size="1.7b",
        hf_model_id="Qwen/Qwen3-1.7B",
        transcoder_repo="mwhanna/qwen3-1.7b-transcoders-lowl0",
        family="qwen3",
    ),
    "qwen3-4b": ModelSpec(
        size="4b",
        hf_model_id="Qwen/Qwen3-4B",
        transcoder_repo="mwhanna/qwen3-4b-transcoders",
        family="qwen3",
    ),
    "qwen3-8b": ModelSpec(
        size="8b",
        hf_model_id="Qwen/Qwen3-8B",
        transcoder_repo="mwhanna/qwen3-8b-transcoders",
        family="qwen3",
    ),
    "qwen3-14b": ModelSpec(
        size="14b",
        hf_model_id="Qwen/Qwen3-14B",
        transcoder_repo="mwhanna/qwen3-14b-transcoders-lowl0",
        family="qwen3",
    ),
    "gemma2-2b": ModelSpec(
        size="2b",
        hf_model_id="google/gemma-2-2b",
        transcoder_repo="mwhanna/gemma-scope-transcoders",
        family="gemma2",
    ),
    "gemma2-2b-cross-layer-426k": ModelSpec(
        size="2b",
        hf_model_id="google/gemma-2-2b",
        transcoder_repo="mntss/clt-gemma-2-2b-426k",
        family="gemma2",
        transcoder_type="cross-layer",
    ),
    "gemma2-2b-cross-layer-2.5m": ModelSpec(
        size="2b",
        hf_model_id="google/gemma-2-2b",
        transcoder_repo="mntss/clt-gemma-2-2b-2.5M",
        family="gemma2",
        transcoder_type="cross-layer",
    ),
}

# Backward-compat aliases: bare size keys resolve to Qwen3 entries.
for _key, _spec in list(_REGISTRY.items()):
    if _spec.family == "qwen3":
        _REGISTRY.setdefault(_spec.size, _spec)


def get_spec(key: str) -> ModelSpec:
    """Look up a ``ModelSpec`` by registry key (e.g. ``"qwen3-0.6b"``, ``"gemma2-2b"``).

    Bare size keys like ``"0.6b"`` still work and resolve to the Qwen3 entry
    for backward compatibility.

    Raises ``KeyError`` with a helpful message if the key is unknown.
    """
    normalised = key.lower().strip()
    if normalised not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown model key {key!r}. Available: {available}")
    return _REGISTRY[normalised]


def list_specs(*, family: str | None = None) -> list[ModelSpec]:
    """Return registered ``ModelSpec`` entries, optionally filtered by family.

    Deduplicates specs so alias entries are not repeated.
    """
    if family is not None:
        family = family.lower()
    unique: list[ModelSpec] = list(
        dict.fromkeys(
            spec for spec in _REGISTRY.values() if family is None or spec.family == family
        )
    )
    return unique


def list_families() -> list[str]:
    """Return sorted list of distinct model families in the registry."""
    return sorted({spec.family for spec in _REGISTRY.values()})


def list_registry() -> list[tuple[str, ModelSpec]]:
    """Return canonical (family-prefixed) key-spec pairs only.

    Excludes bare-size backward-compat aliases so each spec appears once.
    """
    return [(key, spec) for key, spec in _REGISTRY.items() if key.startswith(spec.family)]
