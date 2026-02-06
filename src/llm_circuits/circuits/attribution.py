"""Attribution graph construction.

This module will house our own attribution-graph implementation,
independent of circuit-tracer's ``AttributionGraph``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AttributionNode:
    """A node in the attribution graph."""

    name: str
    layer: int
    node_type: str  # e.g. "mlp", "attn", "transcoder_feature", "residual"
    attribution_score: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class AttributionEdge:
    """A directed edge between two attribution nodes."""

    source: str
    target: str
    weight: float = 0.0


@dataclass
class AttributionGraph:
    """Container for a full attribution graph.

    Nodes represent components (MLP outputs, attention heads, transcoder
    features, etc.) and edges represent the attribution flow between them.
    """

    nodes: list[AttributionNode] = field(default_factory=list)
    edges: list[AttributionEdge] = field(default_factory=list)

    def add_node(self, node: AttributionNode) -> None:
        self.nodes.append(node)

    def add_edge(self, edge: AttributionEdge) -> None:
        self.edges.append(edge)

    def node_names(self) -> list[str]:
        return [n.name for n in self.nodes]
