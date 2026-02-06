"""Load transcoders via circuit-tracer's HF utilities.

This module is the **only** place in llm-circuits that imports from
``circuit_tracer``.  We deliberately avoid importing ``ReplacementModel``
or any attribution-graph machinery -- those will be reimplemented under
``llm_circuits.circuits``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm_circuits.logging import get_logger
from llm_circuits.transcoders.registry import ModelSpec, get_spec

log = get_logger(__name__)


@dataclass
class LoadedTranscoder:
    """Lightweight container returned by :func:`load_transcoder`."""

    transcoder: Any
    """The transcoder object returned by circuit-tracer."""

    config: dict = field(default_factory=dict)
    """Config dict attached to the transcoder, if available."""

    repo_id: str = ""
    """The HF repo the transcoder was loaded from."""


def load_transcoder(
    spec_or_repo: str | ModelSpec,
    *,
    device: str | None = None,
    dtype: Any | None = None,
    lazy_decoder: bool = True,
    lazy_encoder: bool = False,
) -> LoadedTranscoder:
    """Download (or use cached) transcoders and return a :class:`LoadedTranscoder`.

    Args:
        spec_or_repo: Either a size key (e.g. ``"0.6b"``), a :class:`ModelSpec`,
            or a raw HF repo id string like ``"mwhanna/qwen3-0.6b-transcoders-lowl0"``.
        device: Torch device. ``None`` uses the default from settings.
        dtype: Torch dtype override.
        lazy_decoder: If ``True``, decoder weights are loaded lazily.
        lazy_encoder: If ``True``, encoder weights are loaded lazily.

    Returns:
        A :class:`LoadedTranscoder` wrapping the circuit-tracer result.
    """
    from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

    # Resolve the repo id.
    if isinstance(spec_or_repo, ModelSpec):
        repo_id = spec_or_repo.transcoder_repo
    else:
        # Try as a registry key first, then fall back to treating it as a repo id.
        try:
            spec = get_spec(spec_or_repo)
            repo_id = spec.transcoder_repo
        except KeyError:
            repo_id = spec_or_repo

    kwargs: dict[str, Any] = {}
    if device is not None:
        kwargs["device"] = device
    if dtype is not None:
        kwargs["dtype"] = dtype
    kwargs["lazy_decoder"] = lazy_decoder
    kwargs["lazy_encoder"] = lazy_encoder

    log.info("Loading transcoders from [bold]%s[/bold]", repo_id)
    transcoder = load_transcoder_from_hub(repo_id, **kwargs)

    # Try to extract config if the returned object exposes one.
    config: dict = {}
    if hasattr(transcoder, "config"):
        cfg = transcoder.config
        config = (
            dict(cfg) if isinstance(cfg, dict) else vars(cfg) if hasattr(cfg, "__dict__") else {}
        )

    return LoadedTranscoder(transcoder=transcoder, config=config, repo_id=repo_id)


def cache_transcoder(repo_id: str, cache_dir: str) -> None:
    """Download and cache transcoder weights to a local directory.

    Args:
        repo_id: HF repo id for the transcoder, e.g.
            ``"mwhanna/qwen3-0.6b-transcoders-lowl0"``.
        cache_dir: Local directory to save cached files.
    """
    from circuit_tracer.utils.caching import save_transcoders_to_cache

    log.info("Caching transcoders from %s -> %s", repo_id, cache_dir)
    save_transcoders_to_cache(repo_id, cache_dir)
