"""Tests for the HTML renderers (CPU, no model/network)."""

from __future__ import annotations

from llm_circuits.circuits.visualization import (
    render_graph_html,
    render_graph_html_str,
    render_suite_html,
)

_GRAPH = {
    "prompt": "2+3=",
    "tokens": ["a", "b"],
    "logit_token_strs": {"5": "5"},
    "nodes": [
        {"node_type": "embedding", "layer": -1, "position": 0, "activation": 1.0, "label": None},
        {
            "node_type": "feature",
            "layer": 1,
            "position": 1,
            "feature_idx": 7,
            "activation": 2.0,
            "label": {"top_logits": ["five"], "bottom_logits": [], "activation_frequency": 0.1},
        },
        {
            "node_type": "logit",
            "layer": 2,
            "position": 1,
            "token_id": 5,
            "activation": 3.0,
            "label": None,
        },
    ],
    "edges": [
        {"source": 0, "target": 1, "weight": 1.0},
        {"source": 1, "target": 2, "weight": 2.0},
    ],
}


class TestRenderGraphHtmlStr:
    def test_contains_info_panel_and_labels(self):
        s = render_graph_html_str(_GRAPH)
        assert 'id="info"' in s  # click-to-pin panel
        assert "const nodeLabels" in s  # short labels for connections
        assert "five" in s  # feature label embedded
        assert "Top logits" in s  # tooltip content for the labeled feature

    def test_writes_file(self, tmp_path):
        out = render_graph_html(_GRAPH, tmp_path / "g.html")
        assert out.exists()
        assert "<svg" in out.read_text()


class TestRenderSuiteHtml:
    def test_multi_example_viewer(self, tmp_path):
        s = render_graph_html_str(_GRAPH)
        entries = [
            {"label": "2+3=5", "summary": "<b>x</b>", "graph_html": s},
            {"label": "4+5=9", "summary": "<b>y</b>", "graph_html": s},
        ]
        out = render_suite_html(entries, tmp_path / "suite.html", title="Suite")
        t = out.read_text()
        assert "<select" in t
        assert t.count('class="gframe"') == 2
        assert "srcdoc=" in t
        assert "2+3=5" in t and "4+5=9" in t
        # iframe content must be attribute-escaped (no raw <!DOCTYPE leaking out)
        assert "&lt;!DOCTYPE" in t

    def test_empty_raises(self, tmp_path):
        import pytest

        with pytest.raises(ValueError, match="at least one entry"):
            render_suite_html([], tmp_path / "x.html")
