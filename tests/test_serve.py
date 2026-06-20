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
        json={"nodes": [{"layer": 10, "feature_idx": 7, "position": 1}], "factor": -1.0},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["baseline"] and body["steered"]
    assert all("prob" in t and "token" in t for t in body["baseline"])


def test_sweep_returns_curve(client):
    client.post("/api/load", json={"size": "4b"})
    r = client.post(
        "/api/sweep",
        json={"nodes": [{"layer": 10, "feature_idx": 7, "position": 1}], "factor": -1.0},
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["end_layers"]) == len(body["delta_logits"]) == len(body["probs"])
    assert body["best_end_layer"] in body["end_layers"]
