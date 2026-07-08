"""Tests for llm_circuits.settings device resolution (no network / GPU required)."""

from __future__ import annotations

import importlib.util

import pytest
import torch

from llm_circuits.settings import default_device, default_dtype

_HAS_TORCH_XLA = importlib.util.find_spec("torch_xla") is not None


class TestDefaultDevice:
    def test_env_override_passthrough(self, monkeypatch):
        monkeypatch.setenv("LLM_CIRCUITS_DEVICE", "cpu")
        assert default_device() == "cpu"

    def test_no_override_returns_known_device(self, monkeypatch):
        monkeypatch.delenv("LLM_CIRCUITS_DEVICE", raising=False)
        assert default_device() in ("cuda", "mps", "cpu")

    @pytest.mark.skipif(_HAS_TORCH_XLA, reason="torch_xla installed; tpu resolves to xla")
    @pytest.mark.parametrize("value", ["tpu", "TPU", "xla", " tpu "])
    def test_tpu_without_torch_xla_raises(self, monkeypatch, value):
        monkeypatch.setenv("LLM_CIRCUITS_DEVICE", value)
        with pytest.raises(RuntimeError, match=r"torch_xla.*gcp/bootstrap\.sh"):
            default_device()

    @pytest.mark.skipif(not _HAS_TORCH_XLA, reason="torch_xla not installed")
    def test_tpu_resolves_to_xla(self, monkeypatch):
        monkeypatch.setenv("LLM_CIRCUITS_DEVICE", "tpu")
        assert default_device() == "xla"


class TestDefaultDtype:
    def test_cpu_is_bf16(self, monkeypatch):
        monkeypatch.setenv("LLM_CIRCUITS_DEVICE", "cpu")
        assert default_dtype() == torch.bfloat16

    def test_mps_is_fp32(self, monkeypatch):
        monkeypatch.setenv("LLM_CIRCUITS_DEVICE", "mps")
        assert default_dtype() == torch.float32
