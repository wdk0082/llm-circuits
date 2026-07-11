"""CPU tests for the interactive server via the MockEngine (no model)."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from llm_circuits.serve.app import app, set_engine
from llm_circuits.serve.engine import MockEngine


@pytest.fixture
def client():
    set_engine(MockEngine())  # force mock regardless of CUDA availability
    return TestClient(app)


def test_models_lists_qwen3_sizes(client):
    r = client.get("/api/models")
    assert r.status_code == 200
    body = r.json()
    assert "4b" in body["sizes"] and body["mock"] is True
    assert body["loaded"] is None


def test_build_requires_load(client):
    # build before load -> 409
    r = client.post("/api/build", json={"text": "hi"})
    assert r.status_code == 409


def test_load_then_build_returns_explorer_html(client):
    assert client.post("/api/load", json={"size": "4b"}).json()["loaded"] == "4b"
    r = client.post("/api/build", json={"text": "the capital of France is"})
    assert r.status_code == 200
    body = r.json()
    # build returns the self-contained explorer HTML for the iframe + node stats
    assert "<!DOCTYPE html>" in body["html"] and "const D =" in body["html"]
    assert "getGroupsForSteering" in body["html"]  # host page reads supernodes from it
    assert body["n_nodes"] >= 1 and body["n_feature_nodes"] >= 0


def test_build_accepts_influence_node_selection(client):
    # circuit-tracer-aligned cap criterion + no-cap are accepted by the API contract
    client.post("/api/load", json={"size": "4b"})
    r = client.post(
        "/api/build",
        json={"text": "hi", "node_selection": "influence", "max_feature_nodes": None},
    )
    assert r.status_code == 200


def test_build_too_dense_maps_to_400(client, monkeypatch):
    # The too-dense-prompt guard raises ValueError -> the route must surface it as an
    # actionable 400 (not a bare 500) so the user sees the "set a feat-nodes cap" hint.
    import llm_circuits.serve.app as app_mod

    client.post("/api/load", json={"size": "4b"})

    def boom(_req):
        raise ValueError("too many active feature nodes; set a feat-nodes cap")

    monkeypatch.setattr(app_mod._engine, "build", boom)
    r = client.post("/api/build", json={"text": "x", "max_feature_nodes": None})
    assert r.status_code == 400
    assert "feat-nodes cap" in r.json()["detail"]


def test_load_unknown_size_400(client):
    r = client.post("/api/load", json={"size": "999b"})
    assert r.status_code == 400


def test_reprune_requires_build(client):
    client.post("/api/load", json={"size": "4b"})
    r = client.post("/api/reprune", json={"node_threshold": 0.9, "edge_threshold": 0.98})
    assert r.status_code == 409  # must build first


def test_reprune_after_build(client):
    client.post("/api/load", json={"size": "4b"})
    client.post("/api/build", json={"text": "hello"})
    r = client.post("/api/reprune", json={"node_threshold": 0.95, "edge_threshold": 0.99})
    assert r.status_code == 200
    assert "<!DOCTYPE html>" in r.json()["html"]


def test_steer_returns_baseline_and_steered(client):
    client.post("/api/load", json={"size": "4b"})
    r = client.post(
        "/api/steer",
        json={"nodes": [{"layer": 10, "feature_idx": 7, "position": 1}], "m": -2.0},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["baseline"] and body["steered"]
    assert all("prob" in t and "token" in t for t in body["baseline"])


def test_sweep_returns_curve(client):
    client.post("/api/load", json={"size": "4b"})
    r = client.post(
        "/api/sweep",
        json={"nodes": [{"layer": 10, "feature_idx": 7, "position": 1}], "m": -2.0},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["end_layers"]) == len(body["delta_logits"]) == len(body["probs"])
    assert body["best_end_layer"] in body["end_layers"]


def test_real_engine_tokenize_raw_prepends_sink_and_counts_bos():
    # RealEngine._tokenize raw mode must follow the notebooks' tokenize_raw discipline:
    # one special token prepended as the attention sink, n_bos=1 (DEVLOG_EXTRA §5 fix).
    from types import SimpleNamespace

    import torch

    from llm_circuits.serve.engine import RealEngine

    class _StubTok:
        bos_token_id = None
        pad_token_id = None
        eos_token_id = 7

        def __call__(self, text, add_special_tokens=True, return_tensors=None):
            assert add_special_tokens is False
            return SimpleNamespace(input_ids=torch.tensor([[11, 12, 13]]))

    eng = RealEngine.__new__(RealEngine)
    eng.tokenizer = _StubTok()
    eng.model = SimpleNamespace(device="cpu")
    ids, n_bos = eng._tokenize("calc: 1+2=", use_chat=False)
    assert ids.tolist() == [[7, 11, 12, 13]]  # eos prepended as the sink
    assert n_bos == 1


def test_real_engine_tokenize_chat_uses_prepare_messages_bos_count():
    from types import SimpleNamespace

    import torch

    from llm_circuits.serve.engine import RealEngine

    class _StubTok:
        def apply_chat_template(
            self, messages, return_tensors=None, add_generation_prompt=True, **kw
        ):
            assert messages[-1]["role"] == "user"
            return torch.tensor([[1, 2, 3]])

    eng = RealEngine.__new__(RealEngine)
    eng.tokenizer = _StubTok()
    eng.model = SimpleNamespace(device="cpu")
    ids, n_bos = eng._tokenize("hi", use_chat=True)
    assert ids.tolist() == [[1, 2, 3]]
    assert n_bos == 1  # prepare_messages("qwen3") reports one BOS-like token
