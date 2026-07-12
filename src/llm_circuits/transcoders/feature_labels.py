"""Load feature labels (top/bottom logits) from HF transcoder repos.

The ``features/`` directory in each transcoder repo contains:

* ``index.json.gz`` — maps layer number (string key) to
  ``{"filename": "layer_N.bin", "offsets": [...]}``.  Offsets are byte
  positions into the ``.bin`` file for each feature index.
* ``layer_N.bin`` — concatenated feature blobs.  Each blob is a 4-byte
  little-endian header (decompressed size) followed by gzip-compressed JSON.

This module downloads only the index and the layer ``.bin`` files that are
actually needed, then seeks directly to the requested feature offsets so
that we never have to load an entire 1 GB+ file into memory.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field

from huggingface_hub import hf_hub_download

from llm_circuits.logging import get_logger
from llm_circuits.settings import cache_dir as llm_cache_dir

log = get_logger(__name__)


@dataclass
class FeatureLabel:
    """Lightweight container for a single transcoder feature's label metadata."""

    feature_idx: int
    top_logits: list[str] = field(default_factory=list)
    bottom_logits: list[str] = field(default_factory=list)
    activation_frequency: float = 0.0
    act_min: float = 0.0
    act_max: float = 0.0
    quantile_values: list[float] = field(default_factory=list)
    histogram: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "feature_idx": self.feature_idx,
            "top_logits": self.top_logits,
            "bottom_logits": self.bottom_logits,
            "activation_frequency": self.activation_frequency,
            "act_min": self.act_min,
            "act_max": self.act_max,
            "quantile_values": self.quantile_values,
            "histogram": self.histogram,
        }


# Module-level cache so the index is only downloaded & parsed once per repo.
_index_cache: dict[str, dict] = {}
_bin_path_cache: dict[tuple[str, int], str] = {}


def _get_index(repo_id: str) -> dict:
    """Download (or return cached) ``features/index.json.gz`` for *repo_id*."""
    if repo_id in _index_cache:
        return _index_cache[repo_id]

    log.info("Downloading feature index from %s ...", repo_id)
    hf_cache = str(llm_cache_dir() / "feature_labels")
    path = hf_hub_download(repo_id, "features/index.json.gz", cache_dir=hf_cache)
    with gzip.open(path, "rt") as f:
        index = json.load(f)
    _index_cache[repo_id] = index
    return index


def _get_bin_path(repo_id: str, layer: int, index: dict) -> str:
    """Download (or return cached) the layer ``.bin`` file path."""
    key = (repo_id, layer)
    if key in _bin_path_cache:
        return _bin_path_cache[key]

    layer_key = str(layer)
    if layer_key not in index:
        raise KeyError(f"Layer {layer} not found in feature index for {repo_id}")

    filename = index[layer_key]["filename"]
    log.info("Downloading feature file %s from %s ...", filename, repo_id)
    hf_cache = str(llm_cache_dir() / "feature_labels")
    local_path = hf_hub_download(repo_id, f"features/{filename}", cache_dir=hf_cache)
    _bin_path_cache[key] = local_path
    return local_path


def _read_feature_blob(bin_path: str, offsets: list[int], feature_idx: int) -> dict:
    """Read and decompress a single feature blob from a ``.bin`` file."""
    if feature_idx + 1 >= len(offsets):
        raise IndexError(f"Feature index {feature_idx} out of range (max {len(offsets) - 2})")
    start = offsets[feature_idx]
    end = offsets[feature_idx + 1]
    if end <= start:
        # Empty blob: the repo stores no label for this feature (0.4% of the 4b features).
        # Absent upstream data, not damaged data — the caller skips it quietly.
        raise KeyError(f"no label stored for feature {feature_idx}")

    with open(bin_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start)

    # First 4 bytes are a little-endian header, then gzip data
    gzip_start = chunk.find(b"\x1f\x8b")
    if gzip_start == -1:
        raise ValueError(f"No gzip magic found in feature blob at index {feature_idx}")
    decompressed = gzip.decompress(chunk[gzip_start:])
    return json.loads(decompressed)


def load_feature_labels(
    repo_id: str,
    layer: int,
    feature_indices: list[int],
) -> dict[int, FeatureLabel]:
    """Load feature labels for specific features in a layer.

    Args:
        repo_id: HF repo id (e.g. ``"mwhanna/qwen3-0.6b-transcoders-lowl0"``).
        layer: Transformer layer index.
        feature_indices: Which feature indices to load labels for.

    Returns:
        Mapping from feature index to :class:`FeatureLabel`.
    """
    index = _get_index(repo_id)
    layer_key = str(layer)
    if layer_key not in index:
        log.warning("No feature data for layer %d in %s", layer, repo_id)
        return {}

    offsets = index[layer_key]["offsets"]
    bin_path = _get_bin_path(repo_id, layer, index)

    labels: dict[int, FeatureLabel] = {}
    for feat_idx in feature_indices:
        try:
            blob = _read_feature_blob(bin_path, offsets, feat_idx)
            labels[feat_idx] = FeatureLabel(
                feature_idx=feat_idx,
                top_logits=blob.get("top_logits", []),
                bottom_logits=blob.get("bottom_logits", []),
                activation_frequency=blob.get("activation_frequency", 0.0),
                act_min=blob.get("act_min", 0.0),
                act_max=blob.get("act_max", 0.0),
                quantile_values=blob.get("quantile_values", []),
                histogram=blob.get("histogram", []),
            )
        except KeyError:  # no label stored upstream — expected, not an error
            log.debug("No label stored for layer %d feature %d", layer, feat_idx)
        except (IndexError, ValueError) as exc:  # genuinely malformed blob
            log.warning("Failed to load label for layer %d feature %d: %s", layer, feat_idx, exc)

    return labels


def _window_example(tokens: list, acts: list, window: int) -> dict:
    """Trim an example to a context window around its peak-activation token."""
    acts_f = [float(a) for a in acts]
    if not acts_f:
        return {"tokens": list(tokens), "acts": []}
    peak = max(range(len(acts_f)), key=lambda j: acts_f[j])
    lo = max(0, peak - window)
    hi = min(len(tokens), peak + window + 1)
    return {
        "tokens": list(tokens[lo:hi]),
        "acts": [round(a, 3) for a in acts_f[lo:hi]],
    }


def load_feature_examples(
    repo_id: str,
    layer: int,
    feature_indices: list[int],
    *,
    n_per_quantile: int = 5,
    window: int = 14,
) -> dict[int, list[dict]]:
    """Load max-activating dataset examples for features, grouped by quantile.

    Returns ``{feature_idx: [{"quantile": name, "items": [{"tokens", "acts"}]}]}``
    covering **every** quantile the transcoder provides (``Top``, the subsample
    intervals, ``Bottom``) -- not just the top one -- so the explorer can show the
    full activation-example spectrum.  Each example is trimmed to a *window* of
    tokens on either side of its peak-activation token (``window`` each side) and
    capped to *n_per_quantile* examples per quantile.
    """
    index = _get_index(repo_id)
    layer_key = str(layer)
    if layer_key not in index:
        return {}

    offsets = index[layer_key]["offsets"]
    bin_path = _get_bin_path(repo_id, layer, index)

    out: dict[int, list[dict]] = {}
    for feat_idx in feature_indices:
        try:
            blob = _read_feature_blob(bin_path, offsets, feat_idx)
        except (KeyError, IndexError, ValueError):  # KeyError: no blob stored for this feature
            continue
        grouped: list[dict] = []
        for q in blob.get("examples_quantiles", []):
            name = str(q.get("quantile_name", "?"))
            items = [
                _window_example(ex.get("tokens", []), ex.get("tokens_acts_list", []), window)
                for ex in q.get("examples", [])[:n_per_quantile]
            ]
            if items:
                grouped.append({"quantile": name, "items": items})
        out[feat_idx] = grouped
    return out
