"""Interactive HTML visualization for attribution graphs.

Generates a self-contained HTML file with inline SVG and minimal JavaScript.
No external dependencies beyond the Python standard library.

Usage::

    from llm_circuits.circuits.visualization import render_graph_html

    render_graph_html(pruned_dict, "graph.html")
"""

from __future__ import annotations

import html
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Visual constants
# ---------------------------------------------------------------------------

_NODE_COLORS: dict[str, str] = {
    "embedding": "#4CAF50",
    "feature": "#2196F3",
    "error": "#9E9E9E",
    "logit": "#FF9800",
}

_MIN_RADIUS = 8
_MAX_RADIUS = 20
_MIN_EDGE_WIDTH = 0.5
_MAX_EDGE_WIDTH = 4.0
_MARGIN = 60

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def _compute_layout(
    nodes: list[dict],
    width: int,
    height: int,
) -> list[dict[str, float]]:
    """Compute ``(x, y)`` for each node using a layered DAG layout."""
    # Group node indices by layer
    layers: dict[int, list[int]] = {}
    for i, nd in enumerate(nodes):
        layers.setdefault(nd["layer"], []).append(i)

    sorted_layer_keys = sorted(layers.keys())
    n_layers = len(sorted_layer_keys)
    if n_layers == 0:
        return []

    layer_to_x: dict[int, float] = {}
    if n_layers == 1:
        layer_to_x[sorted_layer_keys[0]] = width / 2
    else:
        for rank, key in enumerate(sorted_layer_keys):
            layer_to_x[key] = _MARGIN + rank * (width - 2 * _MARGIN) / (n_layers - 1)

    positions: list[dict[str, float]] = [{"x": 0.0, "y": 0.0}] * len(nodes)
    for layer_key in sorted_layer_keys:
        indices = layers[layer_key]
        # Sort within layer by sequence position for consistent ordering
        indices.sort(key=lambda i: nodes[i].get("position", 0))
        n = len(indices)
        x = layer_to_x[layer_key]
        for rank, idx in enumerate(indices):
            y = height / 2 if n == 1 else _MARGIN + rank * (height - 2 * _MARGIN) / (n - 1)
            positions[idx] = {"x": x, "y": y}

    return positions


# ---------------------------------------------------------------------------
# Node helpers
# ---------------------------------------------------------------------------


def _node_color(node_type: str) -> str:
    return _NODE_COLORS.get(node_type, "#888888")


def _node_radius(activation: float, act_min: float, act_range: float) -> float:
    if act_range == 0:
        return (_MIN_RADIUS + _MAX_RADIUS) / 2
    t = (abs(activation) - act_min) / act_range
    return _MIN_RADIUS + t * (_MAX_RADIUS - _MIN_RADIUS)


def _node_display_label(node: dict, tokens: list[str] | None) -> str:
    ntype = node["node_type"]
    if ntype == "embedding":
        if tokens and 0 <= node["position"] < len(tokens):
            return tokens[node["position"]]
        return f"p{node['position']}"
    if ntype == "feature":
        return f"f{node['feature_idx']}"
    if ntype == "error":
        return "err"
    if ntype == "logit":
        if tokens and node.get("token_id") is not None:
            # token_id doesn't directly index into tokens list; just show id
            return f"tok{node['token_id']}"
        return "logit"
    return "?"


def _node_tooltip_html(node: dict, tokens: list[str] | None) -> str:
    ntype = node["node_type"]
    layer = node["layer"]
    pos = node["position"]
    lines: list[str] = []

    # Header
    if ntype == "embedding":
        tok = ""
        if tokens and 0 <= pos < len(tokens):
            tok = f" &ldquo;{html.escape(tokens[pos])}&rdquo;"
        lines.append(f"<b>Embedding</b> pos={pos}{tok}")
    elif ntype == "feature":
        lines.append(f"<b>Feature {node['feature_idx']}</b> (Layer {layer}, Pos {pos})")
    elif ntype == "error":
        lines.append(f"<b>Error</b> (Layer {layer}, Pos {pos})")
    elif ntype == "logit":
        lines.append(f"<b>Logit</b> tok={node.get('token_id')} (Pos {pos})")

    # Activation & influence
    act = node.get("activation", 0.0)
    lines.append(f"Activation: {act:.4f}")
    inf = node.get("influence")
    if inf is not None:
        lines.append(f"Influence: {inf:.6f}")

    # Feature label
    label = node.get("label")
    if label and isinstance(label, dict):
        top = label.get("top_logits")
        if top:
            top_str = ", ".join(f"&ldquo;{html.escape(str(t))}&rdquo;" for t in top[:8])
            lines.append(f"Top logits: {top_str}")
        bot = label.get("bottom_logits")
        if bot:
            bot_str = ", ".join(f"&ldquo;{html.escape(str(t))}&rdquo;" for t in bot[:5])
            lines.append(f"Bottom logits: {bot_str}")
        freq = label.get("activation_frequency")
        if freq is not None:
            lines.append(f"Freq: {freq:.4f}")

    return "<br>".join(lines)


# ---------------------------------------------------------------------------
# Edge helpers
# ---------------------------------------------------------------------------


def _edge_width(weight: float, max_abs: float) -> float:
    if max_abs == 0:
        return 1.0
    t = abs(weight) / max_abs
    return _MIN_EDGE_WIDTH + t * (_MAX_EDGE_WIDTH - _MIN_EDGE_WIDTH)


def _edge_color(weight: float, max_abs: float) -> str:
    alpha = 0.3 if max_abs == 0 else 0.15 + 0.65 * (abs(weight) / max_abs)
    alpha = min(alpha, 0.8)
    if weight >= 0:
        return f"rgba(46,125,50,{alpha:.2f})"
    return f"rgba(198,40,40,{alpha:.2f})"


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def _render_html(
    nodes: list[dict],
    edges: list[dict],
    layout: list[dict[str, float]],
    tokens: list[str] | None,
    title: str,
    width: int,
    height: int,
) -> str:
    # Pre-compute activation range for radius scaling
    activations = [abs(nd.get("activation", 0.0)) for nd in nodes]
    act_min = min(activations) if activations else 0
    act_max = max(activations) if activations else 0
    act_range = act_max - act_min

    # Pre-compute max edge weight for scaling
    abs_weights = [abs(e["weight"]) for e in edges] if edges else [0]
    max_abs_w = max(abs_weights) if abs_weights else 0

    # --- SVG edges ---
    edge_lines: list[str] = []
    for i, e in enumerate(edges):
        sx = layout[e["source"]]["x"]
        sy = layout[e["source"]]["y"]
        tx = layout[e["target"]]["x"]
        ty = layout[e["target"]]["y"]
        w = _edge_width(e["weight"], max_abs_w)
        c = _edge_color(e["weight"], max_abs_w)
        edge_lines.append(
            f'<line class="edge" data-idx="{i}" data-src="{e["source"]}" '
            f'data-tgt="{e["target"]}" '
            f'x1="{sx:.1f}" y1="{sy:.1f}" x2="{tx:.1f}" y2="{ty:.1f}" '
            f'stroke="{c}" stroke-width="{w:.2f}" />'
        )

    # --- SVG nodes ---
    node_groups: list[str] = []
    for i, nd in enumerate(nodes):
        x = layout[i]["x"]
        y = layout[i]["y"]
        r = _node_radius(nd.get("activation", 0.0), act_min, act_range)
        color = _node_color(nd["node_type"])
        label = html.escape(_node_display_label(nd, tokens))
        # Truncate long labels
        if len(label) > 12:
            label = label[:10] + ".."
        node_groups.append(
            f'<g class="node" data-idx="{i}">'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color}" '
            f'stroke="#333" stroke-width="1.5" />'
            f'<text x="{x:.1f}" y="{y + r + 14:.1f}" text-anchor="middle" '
            f'font-size="11" fill="#333">{label}</text>'
            f"</g>"
        )

    edges_svg = "\n    ".join(edge_lines)
    nodes_svg = "\n    ".join(node_groups)

    # Build tooltip data as JSON for JS
    tooltip_data = []
    for nd in nodes:
        tooltip_data.append(_node_tooltip_html(nd, tokens))

    # Build edge adjacency for highlight: node_idx -> list of edge indices
    node_edges: dict[int, list[int]] = {}
    for i, e in enumerate(edges):
        node_edges.setdefault(e["source"], []).append(i)
        node_edges.setdefault(e["target"], []).append(i)

    title_escaped = html.escape(title)
    tooltips_json = json.dumps(tooltip_data)
    node_edges_json = json.dumps(node_edges)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title_escaped}</title>
<style>
  body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #fafafa; }}
  h2 {{ margin: 12px 16px 4px; font-size: 16px; color: #333; }}
  svg {{ display: block; margin: 0 auto; background: #fff; border: 1px solid #ddd; cursor: grab; }}
  svg:active {{ cursor: grabbing; }}
  .node circle {{ cursor: pointer; transition: opacity 0.15s; }}
  .node text {{ pointer-events: none; user-select: none; }}
  .edge {{ pointer-events: none; }}
  .edge.dim {{ opacity: 0.07; }}
  .edge.highlight {{ opacity: 1 !important; }}
  .node.dim circle {{ opacity: 0.2; }}
  .node.dim text {{ opacity: 0.2; }}
  #tooltip {{
    display: none; position: fixed; padding: 8px 12px; background: rgba(30,30,30,0.92);
    color: #eee; border-radius: 6px; font-size: 12px; line-height: 1.5;
    max-width: 360px; pointer-events: none; z-index: 100; box-shadow: 0 2px 8px rgba(0,0,0,0.3);
  }}
  #legend {{
    position: fixed; bottom: 16px; right: 16px; background: rgba(255,255,255,0.95);
    border: 1px solid #ccc; border-radius: 6px; padding: 10px 14px; font-size: 12px;
  }}
  #legend .item {{ display: flex; align-items: center; margin: 3px 0; }}
  #legend .swatch {{ width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; border: 1px solid #999; }}
  #stats {{ margin: 2px 16px 8px; font-size: 12px; color: #777; }}
</style>
</head>
<body>
<h2>{title_escaped}</h2>
<div id="stats">{len(nodes)} nodes, {len(edges)} edges</div>
<svg id="graph" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <g id="edges">
    {edges_svg}
  </g>
  <g id="nodes">
    {nodes_svg}
  </g>
</svg>
<div id="tooltip"></div>
<div id="legend">
  <div class="item"><div class="swatch" style="background:#4CAF50"></div>Embedding</div>
  <div class="item"><div class="swatch" style="background:#2196F3"></div>Feature</div>
  <div class="item"><div class="swatch" style="background:#9E9E9E"></div>Error</div>
  <div class="item"><div class="swatch" style="background:#FF9800"></div>Logit</div>
  <div style="margin-top:6px; border-top:1px solid #ddd; padding-top:6px;">
    <div class="item" style="font-size:11px;color:#555;">
      <span style="color:rgba(46,125,50,0.8);margin-right:4px;">&#x2500;&#x2500;</span> positive
      &nbsp;
      <span style="color:rgba(198,40,40,0.8);margin-right:4px;">&#x2500;&#x2500;</span> negative
    </div>
  </div>
</div>
<script>
(function() {{
  const tooltips = {tooltips_json};
  const nodeEdges = {node_edges_json};
  const svg = document.getElementById('graph');
  const tip = document.getElementById('tooltip');
  const allNodes = svg.querySelectorAll('.node');
  const allEdges = svg.querySelectorAll('.edge');

  // Tooltip + highlight on hover
  allNodes.forEach(g => {{
    const idx = parseInt(g.dataset.idx);
    g.addEventListener('mouseenter', e => {{
      tip.innerHTML = tooltips[idx];
      tip.style.display = 'block';
      // Dim everything, highlight connected
      allEdges.forEach(el => el.classList.add('dim'));
      allNodes.forEach(el => el.classList.add('dim'));
      g.classList.remove('dim');
      (nodeEdges[idx] || []).forEach(ei => {{
        const el = allEdges[ei];
        if (el) {{ el.classList.remove('dim'); el.classList.add('highlight'); }}
        // Also highlight the other node
        const src = parseInt(el.dataset.src), tgt = parseInt(el.dataset.tgt);
        const other = src === idx ? tgt : src;
        allNodes[other] && allNodes[other].classList.remove('dim');
      }});
    }});
    g.addEventListener('mousemove', e => {{
      tip.style.left = (e.clientX + 14) + 'px';
      tip.style.top = (e.clientY + 14) + 'px';
    }});
    g.addEventListener('mouseleave', () => {{
      tip.style.display = 'none';
      allEdges.forEach(el => {{ el.classList.remove('dim'); el.classList.remove('highlight'); }});
      allNodes.forEach(el => el.classList.remove('dim'));
    }});
  }});

  // Zoom & pan
  let vb = {{ x: 0, y: 0, w: {width}, h: {height} }};
  let isPanning = false, panStart = {{ x: 0, y: 0 }};
  function setViewBox() {{
    svg.setAttribute('viewBox', vb.x+' '+vb.y+' '+vb.w+' '+vb.h);
  }}
  svg.addEventListener('wheel', e => {{
    e.preventDefault();
    const scale = e.deltaY > 0 ? 1.1 : 0.9;
    const pt = svg.getBoundingClientRect();
    const mx = (e.clientX - pt.left) / pt.width;
    const my = (e.clientY - pt.top) / pt.height;
    const nw = vb.w * scale, nh = vb.h * scale;
    vb.x += (vb.w - nw) * mx;
    vb.y += (vb.h - nh) * my;
    vb.w = nw; vb.h = nh;
    setViewBox();
  }}, {{ passive: false }});
  svg.addEventListener('mousedown', e => {{
    if (e.target === svg || e.target.tagName === 'line') {{
      isPanning = true; panStart = {{ x: e.clientX, y: e.clientY }};
    }}
  }});
  window.addEventListener('mousemove', e => {{
    if (!isPanning) return;
    const dx = (e.clientX - panStart.x) * (vb.w / svg.clientWidth);
    const dy = (e.clientY - panStart.y) * (vb.h / svg.clientHeight);
    vb.x -= dx; vb.y -= dy;
    panStart = {{ x: e.clientX, y: e.clientY }};
    setViewBox();
  }});
  window.addEventListener('mouseup', () => {{ isPanning = false; }});
}})();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_graph_html(
    graph_dict: dict,
    output_path: str | Path,
    *,
    title: str | None = None,
    width: int = 1200,
    height: int = 800,
) -> Path:
    """Render a pruned attribution graph as a self-contained HTML file.

    Parameters
    ----------
    graph_dict:
        JSON-compatible dict as produced by ``graph_to_dict()`` or
        ``prune_graph_dict()``.
    output_path:
        Where to write the HTML file.
    title:
        Page title.  Defaults to ``graph_dict["prompt"]`` if present.
    width, height:
        SVG viewport dimensions in pixels.

    Returns
    -------
    Path
        The resolved output path.
    """
    output_path = Path(output_path)
    nodes = graph_dict.get("nodes", [])
    edges = graph_dict.get("edges", [])
    tokens: list[str] | None = graph_dict.get("tokens")

    if title is None:
        title = graph_dict.get("prompt", "Attribution Graph")

    layout = _compute_layout(nodes, width, height)
    html_str = _render_html(nodes, edges, layout, tokens, title, width, height)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_str, encoding="utf-8")
    return output_path
