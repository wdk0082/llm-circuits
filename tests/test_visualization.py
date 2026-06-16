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


class TestRenderSteeringExplorer:
    def test_explorer_html(self, tmp_path):
        from llm_circuits.circuits.visualization import render_steering_explorer_html

        data = {
            "model": "qwen3-4b",
            "examples": [
                {
                    "label": "2+3=5",
                    "answer": "5",
                    "factors": [1.0, 0.0, -1.0],
                    "tokens": ["5", "6", "4"],
                    "probs": [[0.9, 0.05, 0.02], [0.5, 0.3, 0.1], [0.01, 0.6, 0.2]],
                }
            ],
        }
        out = render_steering_explorer_html(data, tmp_path / "se.html", title="T")
        t = out.read_text()
        assert 'id="slider"' in t
        assert "const DATA" in t
        assert "function render" in t
        assert "2+3=5" in t


class TestRenderGraphExplorer:
    def test_explorer_html(self, tmp_path):
        from llm_circuits.circuits.graph_explorer import render_graph_explorer_html

        gd = {
            "prompt": "2+3=",
            "answer_token": "5",
            "tokens": ["a", "b"],
            "logit_token_strs": {"5": "5"},
            "nodes": [
                {
                    "node_type": "embedding",
                    "layer": -1,
                    "position": 0,
                    "activation": 1.0,
                    "label": None,
                },
                {
                    "node_type": "feature",
                    "layer": 1,
                    "position": 1,
                    "feature_idx": 7,
                    "activation": 2.0,
                    "label": {
                        "top_logits": ["five"],
                        "bottom_logits": ["x"],
                        "activation_frequency": 0.0012,
                        "act_min": 0.01,
                        "act_max": 41.0,
                        "histogram": [9, 4, 2, 1],
                        "quantile_values": [0.01, 1.0, 10.0, 41.0],
                        "examples": [
                            {
                                "quantile": "Top",
                                "items": [{"tokens": [" Nov", " 5"], "acts": [0.0, 9.0]}],
                            },
                            {
                                "quantile": "Subsample interval 1",
                                "items": [{"tokens": [" on", " 5"], "acts": [0.0, 3.0]}],
                            },
                        ],
                    },
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
        out = render_graph_explorer_html(gd, tmp_path / "ge.html", title="T")
        t = out.read_text()
        assert "__DATA__" not in t and "__TITLE__" not in t  # placeholders substituted
        assert 'id="g"' in t and 'id="sg"' in t and 'id="detail"' in t  # graph, subgraph, detail
        assert 'id="mk"' in t  # manual-grouping button
        assert "const D =" in t
        assert "five" in t and "Nov" in t  # label + activation examples embedded

        import json
        import re

        data = json.loads(re.search(r"const D = (\{.*?\});\nconst EX", t, re.S).group(1))
        feat = next(n for n in data["examples"][0]["nodes"] if n["t"] == "feature")
        # quantile-grouped activation examples (request 2)
        assert [q["quantile"] for q in feat["ex"]] == ["Top", "Subsample interval 1"]
        # richer feature stats surfaced (request 3)
        assert feat["freq"] == 0.0012 and feat["amax"] == 41.0
        assert feat["hist"] == [9, 4, 2, 1]
        # node fill = group color, member-clickable subgraph, clickable detail rows
        assert "function nodeFill" in t and "function selectNode" in t
        assert "frow nav" in t
        # activation examples colored by SIGN (green positive / red negative)
        assert "46,125,50" in t and "198,40,40" in t
        # draggable supernodes in the subgraph
        assert "subDrag" in t and "supernode" in t and "subUnitsPerPx" in t
        # ungrouped nodes are hollow; type encoded by shape (incl. legend)
        assert 'class="legend"' in t and "logit" in t
        # the full-size SVG rule must be scoped to the graph svgs, not match the
        # inline legend/histogram svgs (regression: bare `svg {width:100%}` blanked the graph)
        assert "#g, #sg {" in t and "svg { display:block" not in t
        # axis labels: layer ticks (emb/output) on y, token ticks on x (rotated)
        ex = data["examples"][0]
        assert {"emb", "output"} <= {q["label"] for q in ex["yticks"]}
        assert {"a", "b"} <= {q["label"] for q in ex["xticks"]}
        assert "function drawAxes" in t and "rotate(-45" in t

    def test_explorer_multi_graph_dropdown(self, tmp_path):
        import json
        import re

        from llm_circuits.circuits.graph_explorer import render_graph_explorer_html

        def _g(prompt, ans, tok_id):
            return {
                "prompt": prompt,
                "answer_token": ans,
                "tokens": ["a", "b"],
                "logit_token_strs": {str(tok_id): ans},
                "nodes": [
                    {"node_type": "embedding", "layer": -1, "position": 0, "activation": 1.0},
                    {
                        "node_type": "logit",
                        "layer": 2,
                        "position": 1,
                        "token_id": tok_id,
                        "activation": 3.0,
                    },
                ],
                "edges": [{"source": 0, "target": 1, "weight": 1.0}],
            }

        graphs = [_g("What is 1+2?", "3", 3), _g("What is 4+4?", "8", 8)]
        out = render_graph_explorer_html(graphs, tmp_path / "ge_multi.html", title="T")
        t = out.read_text()
        assert 'id="pick"' in t and "function loadExample" in t  # dropdown + switcher
        data = json.loads(re.search(r"const D = (\{.*?\});\nconst EX", t, re.S).group(1))
        assert len(data["examples"]) == 2  # both graphs embedded
        assert [e["label"] for e in data["examples"]] == ["1+2=3", "4+4=8"]  # derived labels
