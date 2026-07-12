"""Tests for llm_circuits.transcoders.registry (no network / GPU required)."""

from __future__ import annotations

import pytest

from llm_circuits.transcoders.registry import (
    ModelSpec,
    get_spec,
    list_families,
    list_registry,
    list_specs,
)


class TestModelSpec:
    def test_frozen(self):
        spec = ModelSpec(size="0.6b", hf_model_id="Qwen/Qwen3-0.6B", transcoder_repo="repo")
        with pytest.raises(AttributeError):
            spec.size = "1.7b"  # type: ignore[misc]

    def test_equality(self):
        a = ModelSpec(size="0.6b", hf_model_id="Qwen/Qwen3-0.6B", transcoder_repo="repo")
        b = ModelSpec(size="0.6b", hf_model_id="Qwen/Qwen3-0.6B", transcoder_repo="repo")
        assert a == b

    def test_default_family(self):
        spec = ModelSpec(size="0.6b", hf_model_id="m", transcoder_repo="r")
        assert spec.family == "qwen3"

    def test_default_transcoder_type(self):
        spec = ModelSpec(size="0.6b", hf_model_id="m", transcoder_repo="r")
        assert spec.transcoder_type == "per-layer"


class TestGetSpec:
    @pytest.mark.parametrize(
        "key,expected_model_id",
        [
            # Qwen3 family-prefixed keys
            ("qwen3-0.6b", "Qwen/Qwen3-0.6B"),
            ("qwen3-1.7b", "Qwen/Qwen3-1.7B"),
            ("qwen3-4b", "Qwen/Qwen3-4B"),
            ("qwen3-8b", "Qwen/Qwen3-8B"),
            ("qwen3-14b", "Qwen/Qwen3-14B"),
            # Qwen3 bare size aliases (backward compat)
            ("0.6b", "Qwen/Qwen3-0.6B"),
            ("1.7b", "Qwen/Qwen3-1.7B"),
            ("4b", "Qwen/Qwen3-4B"),
            ("8b", "Qwen/Qwen3-8B"),
            ("14b", "Qwen/Qwen3-14B"),
        ],
    )
    def test_valid_keys(self, key: str, expected_model_id: str):
        spec = get_spec(key)
        assert spec.hf_model_id == expected_model_id

    def test_case_insensitive(self):
        assert get_spec("0.6B") == get_spec("0.6b")

    def test_strips_whitespace(self):
        assert get_spec("  0.6b  ") == get_spec("0.6b")

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError, match="Unknown model key"):
            get_spec("999b")

    def test_unknown_key_message_lists_available(self):
        with pytest.raises(KeyError, match=r"0\.6b"):
            get_spec("nope")

    def test_bare_size_resolves_to_qwen3(self):
        """Bare size keys like '0.6b' should resolve to the Qwen3 entry."""
        spec = get_spec("0.6b")
        assert spec.family == "qwen3"
        assert spec == get_spec("qwen3-0.6b")


class TestListSpecs:
    def test_returns_list(self):
        specs = list_specs()
        assert isinstance(specs, list)
        assert len(specs) == 5

    def test_all_model_spec(self):
        for spec in list_specs():
            assert isinstance(spec, ModelSpec)

    def test_contains_expected_sizes(self):
        sizes = {s.size for s in list_specs()}
        assert sizes == {"0.6b", "1.7b", "4b", "8b", "14b"}

    def test_transcoder_repos_are_populated(self):
        for spec in list_specs():
            assert spec.transcoder_repo
            assert "/" in spec.transcoder_repo

    def test_filter_by_family_qwen3(self):
        specs = list_specs(family="qwen3")
        assert len(specs) == 5
        assert all(s.family == "qwen3" for s in specs)

    def test_filter_by_family_case_insensitive(self):
        assert list_specs(family="Qwen3") == list_specs(family="qwen3")


class TestListFamilies:
    def test_returns_sorted(self):
        families = list_families()
        assert families == ["qwen3"]


class TestListRegistry:
    def test_returns_canonical_keys(self):
        entries = list_registry()
        assert len(entries) == 5
        keys = [k for k, _ in entries]
        # No bare-size aliases
        for key in keys:
            assert "-" in key

    def test_all_keys_start_with_family(self):
        for key, spec in list_registry():
            assert key.startswith(spec.family)


class TestLoadTranscoderCacheDtype:
    def test_first_load_caches_at_requested_dtype(self, monkeypatch):
        """A first (uncached) load must write the disk cache at the REQUESTED dtype —
        circuit-tracer's fp32 default doubles the bf16 mwhanna repos on disk and
        overflowed a 369 GB studio disk at the 4b+8b pair (2026-07-12)."""
        import sys
        from types import ModuleType, SimpleNamespace

        import torch

        import llm_circuits.transcoders.circuit_tracer_loader as loader

        calls = {}
        caching = ModuleType("circuit_tracer.utils.caching")
        caching.is_cached = lambda repo, cache_dir: False
        caching.load_transcoders_from_cache = lambda repo, **kw: (
            SimpleNamespace(),
            {"scan": repo},
        )
        caching.save_transcoders_to_cache = lambda repo, cache, **kw: calls.update(kw)
        utils = ModuleType("circuit_tracer.utils")
        utils.caching = caching
        ct = ModuleType("circuit_tracer")
        ct.utils = utils
        monkeypatch.setitem(sys.modules, "circuit_tracer", ct)
        monkeypatch.setitem(sys.modules, "circuit_tracer.utils", utils)
        monkeypatch.setitem(sys.modules, "circuit_tracer.utils.caching", caching)

        loader.load_transcoder("qwen3-4b", device="cpu", dtype=torch.bfloat16)
        assert calls.get("dtype") is torch.bfloat16

        calls.clear()
        loader.load_transcoder("qwen3-4b", device="cpu")  # no dtype -> fp32 default
        assert "dtype" not in calls
