"""Load transcoders via circuit-tracer's HF utilities.

This module is the **only** place in llm-circuits that imports from
``circuit_tracer``.  We deliberately avoid importing ``ReplacementModel``
or any attribution-graph machinery -- those will be reimplemented under
``llm_circuits.circuits``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

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

    On first load the transcoders are downloaded from the HF Hub and
    automatically saved into the local cache so that subsequent loads
    are served from disk.

    Args:
        spec_or_repo: Either a size key (e.g. ``"0.6b"``), a :class:`ModelSpec`,
            or a raw HF repo id string like ``"mwhanna/qwen3-0.6b-transcoders-lowl0"``.
        device: Torch device. ``None`` uses the default from settings.
        dtype: Torch dtype override. Also the dtype of the on-disk cache written on a
            first load (``None`` keeps circuit-tracer's fp32 default — for the bf16
            mwhanna repos pass ``torch.bfloat16`` to halve the cache losslessly, per
            the CLAUDE.md perf notes).
        lazy_decoder: If ``True``, decoder weights are loaded lazily.
        lazy_encoder: If ``True``, encoder weights are loaded lazily.
        cache_dir: Local directory for cached transcoders. Defaults to
            :func:`~llm_circuits.settings.transcoder_cache_dir`.

    Returns:
        A :class:`LoadedTranscoder` wrapping the circuit-tracer result.
    """
    from circuit_tracer.utils.caching import is_cached, load_transcoders_from_cache

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

    torch_device = torch.device(device) if isinstance(device, str) else device
    torch_dtype = dtype if dtype is not None else torch.float32

    # --- Try local cache first ---------------------------------------------------
    if is_cached(repo_id, resolved_cache_dir):
        log.info(
            "Loading transcoders from cache [bold]%s[/bold] (cache_dir=%s)",
            repo_id,
            resolved_cache_dir,
        )
        transcoder_obj, config = load_transcoders_from_cache(
            repo_id,
            cache_dir=resolved_cache_dir,
            device=torch_device,
            dtype=torch_dtype,
            lazy_encoder=lazy_encoder,
            lazy_decoder=lazy_decoder,
        )
        config = dict(config) if isinstance(config, dict) else {}
        return LoadedTranscoder(transcoder=transcoder_obj, config=config, repo_id=repo_id)

    # --- Not cached: download, cache, then load from cache -----------------------
    # The disk cache is written at the REQUESTED dtype: circuit-tracer's fp32 default
    # doubles the mwhanna bf16 repos on disk for nothing (a pure upcast), and at the
    # 4b+8b pair that overflows a 369 GB studio disk (measured 2026-07-12: fp32 cached
    # 121 GB for the 4b alone where bf16 is ~60 GB).
    log.info(
        "Downloading transcoders [bold]%s[/bold] and caching to %s (dtype=%s)",
        repo_id,
        resolved_cache_dir,
        torch_dtype,
    )
    cache_transcoder(repo_id, cache_dir=resolved_cache_dir, dtype=dtype)

    transcoder_obj, config = load_transcoders_from_cache(
        repo_id,
        cache_dir=resolved_cache_dir,
        device=torch_device,
        dtype=torch_dtype,
        lazy_encoder=lazy_encoder,
        lazy_decoder=lazy_decoder,
    )
    config = dict(config) if isinstance(config, dict) else {}
    return LoadedTranscoder(transcoder=transcoder_obj, config=config, repo_id=repo_id)


def cache_transcoder(
    repo_id: str, cache_dir: str | None = None, *, dtype: Any | None = None
) -> None:
    """Download and cache transcoder weights to a local directory.

    Args:
        repo_id: HF repo id for the transcoder, e.g.
            ``"mwhanna/qwen3-0.6b-transcoders-lowl0"``.
        cache_dir: Local directory to save cached files. Defaults to
            :func:`~llm_circuits.settings.transcoder_cache_dir`.
        dtype: Dtype of the cached safetensors (circuit-tracer's default is
            fp32). The mwhanna Qwen3 repos store bf16 weights, so caching at
            ``torch.bfloat16`` halves the on-disk size losslessly — the fp32
            default is a pure upcast of bf16 source data.
    """
    from circuit_tracer.utils.caching import save_transcoders_to_cache

    from llm_circuits.settings import transcoder_cache_dir

    resolved = cache_dir if cache_dir is not None else str(transcoder_cache_dir())
    log.info("Caching transcoders from %s -> %s", repo_id, resolved)
    kwargs = {} if dtype is None else {"dtype": dtype}
    save_transcoders_to_cache(repo_id, resolved, **kwargs)
