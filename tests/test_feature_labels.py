"""Tests for the feature-label blob reader.

The label repos store one gzip blob per feature in a flat ``.bin``, addressed by an
offsets table. ~0.4% of features have an *empty* slice (offsets[i] == offsets[i+1]):
the repo simply has no label for them. That is absent data, not damaged data, and the
loader must not report it as corruption.
"""

from __future__ import annotations

import gzip
import json

import pytest

from llm_circuits.transcoders.feature_labels import _read_feature_blob


def _write_bin(tmp_path, blobs: list[bytes]):
    """Pack blobs into a .bin + offsets table (an empty bytes entry = no label)."""
    path = tmp_path / "layer_0.bin"
    offsets, buf = [0], bytearray()
    for b in blobs:
        buf += b
        offsets.append(len(buf))
    path.write_bytes(bytes(buf))
    return str(path), offsets


def _blob(payload: dict) -> bytes:
    return b"\x00\x00\x00\x00" + gzip.compress(json.dumps(payload).encode())  # 4-byte header


def test_reads_a_labelled_feature(tmp_path):
    path, offsets = _write_bin(tmp_path, [_blob({"top_logits": ["five", "5"]})])
    assert _read_feature_blob(path, offsets, 0)["top_logits"] == ["five", "5"]


def test_empty_blob_is_missing_data_not_corruption(tmp_path):
    """An unlabelled feature raises KeyError (skipped quietly), never ValueError."""
    path, offsets = _write_bin(tmp_path, [_blob({"top_logits": ["a"]}), b"", _blob({})])
    assert offsets[1] == offsets[2], "feature 1 should have a zero-length slice"

    with pytest.raises(KeyError):
        _read_feature_blob(path, offsets, 1)

    # its neighbours still read fine — the offsets table is not disturbed
    assert _read_feature_blob(path, offsets, 0)["top_logits"] == ["a"]
    assert _read_feature_blob(path, offsets, 2) == {}


def test_malformed_blob_still_raises_valueerror(tmp_path):
    """Non-empty but ungzippable data is real corruption and must stay loud."""
    path, offsets = _write_bin(tmp_path, [b"not gzip at all"])
    with pytest.raises(ValueError, match="No gzip magic"):
        _read_feature_blob(path, offsets, 0)


def test_out_of_range_feature_raises_indexerror(tmp_path):
    path, offsets = _write_bin(tmp_path, [_blob({})])
    with pytest.raises(IndexError):
        _read_feature_blob(path, offsets, 5)


def test_both_public_loaders_swallow_the_unlabelled_case(monkeypatch, tmp_path):
    """Neither loader may propagate the empty-blob KeyError to a caller.

    ``load_feature_examples`` used to catch only (IndexError, ValueError), so making the
    empty blob raise KeyError would have crashed it — and the notebooks call it.
    """
    from llm_circuits.transcoders import feature_labels as fl

    path, offsets = _write_bin(tmp_path, [_blob({"top_logits": ["a"]}), b""])
    index = {"0": {"filename": "layer_0.bin", "offsets": offsets}}
    monkeypatch.setattr(fl, "_get_index", lambda repo_id: index)
    monkeypatch.setattr(fl, "_get_bin_path", lambda repo_id, layer, index: path)

    labels = fl.load_feature_labels("repo", 0, [0, 1])  # feature 1 is unlabelled
    assert sorted(labels) == [0]

    examples = fl.load_feature_examples("repo", 0, [0, 1])
    assert 1 not in examples
