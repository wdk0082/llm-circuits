"""Request/response models for the interactive server API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ModelsResponse(BaseModel):
    sizes: list[str]
    loaded: str | None = None
    n_layers: int | None = None
    mock: bool = False


class LoadRequest(BaseModel):
    size: str


class BuildRequest(BaseModel):
    text: str
    use_chat: bool = False  # wrap in the chat template (instruct Q&A) vs raw completion
    max_feature_targets: int | None = 256
    max_feature_nodes: int | None = 4000  # cap active-feature nodes (dense inputs fire ~1M+)
    node_threshold: float = 0.8
    edge_threshold: float = 0.98


class RepruneRequest(BaseModel):
    """Re-prune the last built graph at new thresholds (fast; no model forward)."""

    node_threshold: float = 0.8
    edge_threshold: float = 0.98


class NodeRef(BaseModel):
    """A feature node to steer (the client sends these from a grouped supernode)."""

    layer: int
    feature_idx: int
    position: int


class SteerRequest(BaseModel):
    nodes: list[NodeRef]
    factor: float = -1.0  # multiplicative M; -1 = negative steer, 0 = ablate, 1 = clean
    mode: str = "constrained"  # or "iterative"
    patch_end_layer: int | None = None
    top_k: int = 10


class SweepRequest(BaseModel):
    nodes: list[NodeRef]
    factor: float = -1.0


class TokenProb(BaseModel):
    token: str
    token_id: int
    prob: float


class BuildResponse(BaseModel):
    html: str  # self-contained explorer HTML (hosted in an <iframe srcdoc>)
    n_nodes: int
    n_feature_nodes: int


class SteerResponse(BaseModel):
    baseline: list[TokenProb]
    steered: list[TokenProb]
    target_token: str = Field(default="")
    target_delta_logit: float = 0.0


class SweepResponse(BaseModel):
    end_layers: list[int]
    delta_logits: list[float]
    probs: list[float]
    best_end_layer: int
    target_token: str = ""
