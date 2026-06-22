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
    # None = compute the FULL edge matrix over the kept feature nodes (cheap now that the
    # edge backward is batched); pruning then selects by influence. Set an int to cap.
    max_feature_targets: int | None = None
    max_feature_nodes: int | None = 4000  # cap active-feature nodes (dense inputs fire ~1M+)
    # How max_feature_nodes selects which features to keep:
    #   "activation" — cheap pre-edge cap by |activation| (fast; the default).
    #   "influence"  — circuit-tracer's criterion: build the full edge matrix over ALL
    #                  active features, then keep the top-N by influence (slower, faithful).
    node_selection: str = "activation"
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
    m: float = 0.0  # additive-delta M: 0 = no change, -1 = ablate, -2 = negative steer
    freeze_attention: bool = True  # freeze attention patterns (forced True when a range is set)
    patch_end_layer: int | None = None
    top_k: int = 10


class SweepRequest(BaseModel):
    nodes: list[NodeRef]
    m: float = -2.0  # negative steer by default (flip)
    freeze_attention: bool = True


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
