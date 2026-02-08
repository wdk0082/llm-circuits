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
    cache_dir: str | None = None,
) -> LoadedTranscoder:
    """Download (or use cached) transcoders and return a :class:`LoadedTranscoder`.

    If transcoders have been previously saved via :func:`cache_transcoder`,
    they will be loaded from the local cache instead of downloading again.

    Args:
        spec_or_repo: Either a size key (e.g. ``"0.6b"``), a :class:`ModelSpec`,
            or a raw HF repo id string like ``"mwhanna/qwen3-0.6b-transcoders-lowl0"``.
        device: Torch device. ``None`` uses the default from settings.
        dtype: Torch dtype override.
        lazy_decoder: If ``True``, decoder weights are loaded lazily.
        lazy_encoder: If ``True``, encoder weights are loaded lazily.
        cache_dir: Local directory to check for cached transcoders. Defaults to
            :func:`~llm_circuits.settings.transcoder_cache_dir`.

    Returns:
        A :class:`LoadedTranscoder` wrapping the circuit-tracer result.
    """
    from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

    from llm_circuits.settings import transcoder_cache_dir

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

    resolved_cache_dir = cache_dir if cache_dir is not None else str(transcoder_cache_dir())

    kwargs: dict[str, Any] = {}
    if device is not None:
        kwargs["device"] = device
    if dtype is not None:
        kwargs["dtype"] = dtype
    kwargs["lazy_decoder"] = lazy_decoder
    kwargs["lazy_encoder"] = lazy_encoder
    kwargs["cache_dir"] = resolved_cache_dir

    log.info("Loading transcoders from [bold]%s[/bold] (cache_dir=%s)", repo_id, resolved_cache_dir)
    transcoder = load_transcoder_from_hub(repo_id, **kwargs)

    # load_transcoder_from_hub returns (transcoder_obj, config_dict).
    config: dict = {}
    if isinstance(transcoder, tuple):
        transcoder, config = transcoder
        config = dict(config) if isinstance(config, dict) else {}
    elif hasattr(transcoder, "config"):
        cfg = transcoder.config
        config = (
            dict(cfg) if isinstance(cfg, dict) else vars(cfg) if hasattr(cfg, "__dict__") else {}
        )

    return LoadedTranscoder(transcoder=transcoder, config=config, repo_id=repo_id)


def cache_transcoder(repo_id: str, cache_dir: str | None = None) -> None:
    """Download and cache transcoder weights to a local directory.

    Args:
        repo_id: HF repo id for the transcoder, e.g.
            ``"mwhanna/qwen3-0.6b-transcoders-lowl0"``.
        cache_dir: Local directory to save cached files. Defaults to
            :func:`~llm_circuits.settings.transcoder_cache_dir`.
    """
    from circuit_tracer.utils.caching import save_transcoders_to_cache

    from llm_circuits.settings import transcoder_cache_dir

    resolved = cache_dir if cache_dir is not None else str(transcoder_cache_dir())
    log.info("Caching transcoders from %s -> %s", repo_id, resolved)
    save_transcoders_to_cache(repo_id, resolved)
