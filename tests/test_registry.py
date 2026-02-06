"""Tests for llm_circuits.transcoders.registry (no network / GPU required)."""

from __future__ import annotations

import pytest

from llm_circuits.transcoders.registry import ModelSpec, get_spec, list_specs


class TestModelSpec:
    def test_frozen(self):
        spec = ModelSpec(size="0.6b", hf_model_id="Qwen/Qwen3-0.6B", transcoder_repo="repo")
        with pytest.raises(AttributeError):
            spec.size = "1.7b"  # type: ignore[misc]

    def test_equality(self):
        a = ModelSpec(size="0.6b", hf_model_id="Qwen/Qwen3-0.6B", transcoder_repo="repo")
        b = ModelSpec(size="0.6b", hf_model_id="Qwen/Qwen3-0.6B", transcoder_repo="repo")
        assert a == b


class TestGetSpec:
    @pytest.mark.parametrize(
        "size,expected_model_id",
        [
            ("0.6b", "Qwen/Qwen3-0.6B"),
            ("1.7b", "Qwen/Qwen3-1.7B"),
            ("4b", "Qwen/Qwen3-4B"),
            ("8b", "Qwen/Qwen3-8B"),
            ("14b", "Qwen/Qwen3-14B"),
        ],
    )
    def test_valid_sizes(self, size: str, expected_model_id: str):
        spec = get_spec(size)
        assert spec.size == size
        assert spec.hf_model_id == expected_model_id

    def test_case_insensitive(self):
        assert get_spec("0.6B") == get_spec("0.6b")

    def test_strips_whitespace(self):
        assert get_spec("  0.6b  ") == get_spec("0.6b")

    def test_unknown_size_raises(self):
        with pytest.raises(KeyError, match="Unknown model size"):
            get_spec("999b")

    def test_unknown_size_message_lists_available(self):
        with pytest.raises(KeyError, match=r"0\.6b"):
            get_spec("nope")


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
