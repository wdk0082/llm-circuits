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

_MIN_RADIUS = 4
_MAX_RADIUS = 12
_MIN_EDGE_WIDTH = 0.5
_MAX_EDGE_WIDTH = 4.0
_MARGIN = 60
_NODE_SPACING = 28  # fixed horizontal spacing between nodes in the same (position, layer) cell

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def _compute_layout(
    nodes: list[dict],
    width: int,
    height: int,
) -> tuple[list[dict[str, float]], int]:
    """Compute ``(x, y)`` for each node.

    X-axis represents sequence **position** and Y-axis represents **layer**
    (embedding at the bottom, logits at the top).  Each position is given a
    column whose width is proportional to the maximum number of nodes at any
    single layer within that position, so busy positions get more room.

    Returns ``(positions, actual_width)`` where *actual_width* is the
    computed width (may exceed the requested *width* to avoid overlap).
    """
    if not nodes:
        return [], width

    # --- Collect distinct positions and layers ---
    sorted_positions = sorted({nd["position"] for nd in nodes})
    sorted_layers = sorted({nd["layer"] for nd in nodes})

    if not sorted_positions or not sorted_layers:
        return [{"x": 0.0, "y": 0.0}] * len(nodes), width

    # --- Y-axis: layer → y (bottom = lowest layer, top = highest) ---
    layer_to_y: dict[int, float] = {}
    n_layers = len(sorted_layers)
    for rank, layer in enumerate(sorted_layers):
        # Invert so lowest layer is at the bottom
        t = rank / (n_layers - 1) if n_layers > 1 else 0.5
        layer_to_y[layer] = height - _MARGIN - t * (height - 2 * _MARGIN)

    # --- X-axis: variable-width position columns, right-aligned nodes ---
    # Group node indices by (position, layer)
    pos_layer_nodes: dict[tuple[int, int], list[int]] = {}
    for i, nd in enumerate(nodes):
        key = (nd["position"], nd["layer"])
        pos_layer_nodes.setdefault(key, []).append(i)

    # Column width = max nodes at any layer in that position * fixed spacing
    pos_max_count: dict[int, int] = {}
    for pos in sorted_positions:
        max_at_layer = 1
        for layer in sorted_layers:
            count = len(pos_layer_nodes.get((pos, layer), []))
            max_at_layer = max(max_at_layer, count)
        pos_max_count[pos] = max_at_layer

    col_widths: dict[int, float] = {
        pos: max(pos_max_count[pos] * _NODE_SPACING, _NODE_SPACING) for pos in sorted_positions
    }
    total_col_width = sum(col_widths.values())
    # Add inter-column gaps
    n_gaps = max(len(sorted_positions) - 1, 0)
    gap = 20.0
    total_needed = total_col_width + n_gaps * gap + 2 * _MARGIN

    # Expand width to fit all columns without overlap
    actual_width = max(width, int(total_needed) + 1)

    # Compute the right edge of each position column
    pos_right: dict[int, float] = {}
    x_cursor = _MARGIN
    for pos in sorted_positions:
        x_cursor += col_widths[pos]
        pos_right[pos] = x_cursor
        x_cursor += gap

    # --- Place nodes (right-aligned, fixed spacing) ---
    positions: list[dict[str, float]] = [{"x": 0.0, "y": 0.0}] * len(nodes)
    for (pos, layer), indices in pos_layer_nodes.items():
        right = pos_right[pos]
        y = layer_to_y[layer]
        n = len(indices)
        # Sort by feature_idx (or token_id) for deterministic ordering
        indices.sort(
            key=lambda i: (nodes[i].get("feature_idx") or 0, nodes[i].get("token_id") or 0)
        )
        # Place nodes right-aligned with fixed spacing
        for rank, idx in enumerate(indices):
            # rightmost node at right edge, others spaced left
            x = right - (n - 1 - rank) * _NODE_SPACING
            positions[idx] = {"x": x, "y": y}

    return positions, actual_width


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


def _make_visible(s: str) -> str:
    """Replace whitespace characters with visible representations."""
    return s.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t").replace(" ", " ")


def _node_display_label(
    node: dict, tokens: list[str] | None, logit_token_strs: dict[str, str] | None = None
) -> str:
    ntype = node["node_type"]
    if ntype == "embedding":
        if tokens and 0 <= node["position"] < len(tokens):
            return _make_visible(tokens[node["position"]])
        return f"p{node['position']}"
    if ntype == "feature":
        return ""  # too many to label; use tooltip on hover
    if ntype == "error":
        return ""
    if ntype == "logit":
        tid = node.get("token_id")
        if tid is not None and logit_token_strs and str(tid) in logit_token_strs:
            return logit_token_strs[str(tid)]
        if tid is not None:
            return f"tok{tid}"
        return "logit"
    return ""


def _node_short_label(
    node: dict, tokens: list[str] | None, logit_token_strs: dict[str, str] | None = None
) -> str:
    """A compact one-line label (HTML-escaped) used in the click-to-pin info panel."""
    ntype = node["node_type"]
    layer = node["layer"]
    pos = node["position"]
    if ntype == "embedding":
        tok = tokens[pos] if (tokens and 0 <= pos < len(tokens)) else f"p{pos}"
        label = f'emb "{_make_visible(tok)}" p{pos}'
    elif ntype == "feature":
        label = f"L{layer} f{node['feature_idx']} p{pos}"
    elif ntype == "error":
        label = f"err L{layer} p{pos}"
    elif ntype == "logit":
        tid = node.get("token_id")
        s = logit_token_strs.get(str(tid)) if (logit_token_strs and tid is not None) else None
        label = f'logit "{s}"' if s else f"logit tok{tid}"
    else:
        label = ntype
    return html.escape(label)


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
    logit_token_strs: dict[str, str] | None,
    title: str,
    width: int,
    height: int,
) -> str:
    # Pre-compute activation range for radius scaling
    activations = [abs(nd.get("activation", 0.0)) for nd in nodes]
    act_min = min(activations) if activations else 0
    act_max = max(activations) if activations else 0
    act_range = act_max - act_min

    # --- SVG nodes only (no edges in initial render) ---
    node_groups: list[str] = []
    for i, nd in enumerate(nodes):
        x = layout[i]["x"]
        y = layout[i]["y"]
        r = _node_radius(nd.get("activation", 0.0), act_min, act_range)
        color = _node_color(nd["node_type"])
        label = html.escape(_node_display_label(nd, tokens, logit_token_strs))
        # No truncation — vertical text has room
        ntype = nd["node_type"]
        # Vertical text: embedding labels below, logit labels above
        if label and ntype == "logit":
            tx = x + 4
            ty = y - r - 6
            text_el = (
                f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="start" '
                f'font-size="11" fill="#333" '
                f'transform="rotate(-90,{tx:.1f},{ty:.1f})">{label}</text>'
            )
        elif label and ntype == "embedding":
            tx = x + 4
            ty = y + r + 6
            text_el = (
                f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="end" '
                f'font-size="11" fill="#333" '
                f'transform="rotate(-90,{tx:.1f},{ty:.1f})">{label}</text>'
            )
        elif label:
            text_el = (
                f'<text x="{x:.1f}" y="{y + r + 14:.1f}" text-anchor="middle" '
                f'font-size="11" fill="#333">{label}</text>'
            )
        else:
            text_el = ""
        node_groups.append(
            f'<g class="node" data-idx="{i}">'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color}" '
            f'stroke="#333" stroke-width="1.5" />'
            f"{text_el}"
            f"</g>"
        )

    nodes_svg = "\n    ".join(node_groups)

    # Build tooltip data as JSON for JS
    tooltip_data = [_node_tooltip_html(nd, tokens) for nd in nodes]
    short_labels = [_node_short_label(nd, tokens, logit_token_strs) for nd in nodes]

    # Build compact edge data for JS: [source, target, weight] per edge
    # and node positions for drawing lines on demand
    node_positions = [
        [round(layout[i]["x"], 1), round(layout[i]["y"], 1)] for i in range(len(nodes))
    ]

    # Build per-node edge index: node_idx -> list of edge indices
    node_edge_map: dict[int, list[int]] = {}
    for i, e in enumerate(edges):
        node_edge_map.setdefault(e["source"], []).append(i)
        node_edge_map.setdefault(e["target"], []).append(i)

    # Compact edge array: [source, target, weight]
    edge_data = [[e["source"], e["target"], round(e["weight"], 6)] for e in edges]

    # Pre-compute max abs weight for JS edge styling
    abs_weights = [abs(e["weight"]) for e in edges] if edges else [0]
    max_abs_w = max(abs_weights) if abs_weights else 0

    title_escaped = html.escape(title)
    tooltips_json = json.dumps(tooltip_data)
    short_labels_json = json.dumps(short_labels)
    node_pos_json = json.dumps(node_positions)
    edge_data_json = json.dumps(edge_data)
    node_edge_map_json = json.dumps(node_edge_map)

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
  .node.dim circle {{ opacity: 0.2; }}
  .node.dim text {{ opacity: 0.2; }}
  #tooltip {{
    display: none; position: fixed; padding: 8px 12px; background: rgba(30,30,30,0.92);
    color: #eee; border-radius: 6px; font-size: 12px; line-height: 1.5;
    max-width: 360px; pointer-events: none; z-index: 100; box-shadow: 0 2px 8px rgba(0,0,0,0.3);
  }}
  #info {{
    display: none; position: fixed; top: 16px; right: 16px; width: 300px; max-height: 78vh;
    overflow: auto; background: #fff; color: #222; border: 1px solid #bbb; border-radius: 6px;
    padding: 10px 12px; font-size: 12px; line-height: 1.55; z-index: 90;
    box-shadow: 0 2px 10px rgba(0,0,0,0.25);
  }}
  #legend {{
    position: fixed; bottom: 16px; right: 16px; background: rgba(255,255,255,0.95);
    border: 1px solid #ccc; border-radius: 6px; padding: 10px 14px; font-size: 12px;
  }}
  #legend .item {{ display: flex; align-items: center; margin: 3px 0; }}
  #legend .swatch {{ width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; border: 1px solid #999; }}
  #stats {{ margin: 2px 16px 8px; font-size: 12px; color: #777; }}
  #hint {{ margin: 2px 16px; font-size: 11px; color: #999; }}
</style>
</head>
<body>
<h2>{title_escaped}</h2>
<div id="stats">{len(nodes)} nodes, {len(edges)} edges</div>
<div id="hint">Click a node to pin its label &amp; connections (panel, top-right). Click it again or the background to close. Hover any node for a quick tooltip.</div>
<svg id="graph" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <g id="edges"></g>
  <g id="nodes">
    {nodes_svg}
  </g>
</svg>
<div id="tooltip"></div>
<div id="info"></div>
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
  <div style="margin-top:4px;font-size:11px;color:#555;">
    Node size = |activation|
  </div>
</div>
<script>
(function() {{
  const tooltips = {tooltips_json};
  const nodeLabels = {short_labels_json};
  const nodePos = {node_pos_json};
  const edgeData = {edge_data_json};
  const nodeEdgeMap = {node_edge_map_json};
  const maxAbsW = {max_abs_w};
  const svg = document.getElementById('graph');
  const tip = document.getElementById('tooltip');
  const info = document.getElementById('info');
  const edgesG = document.getElementById('edges');
  const allNodes = svg.querySelectorAll('.node');
  const SVG_NS = 'http://www.w3.org/2000/svg';

  let selectedIdx = null;

  function edgeColor(w) {{
    const a = maxAbsW === 0 ? 0.3 : 0.15 + 0.65 * (Math.abs(w) / maxAbsW);
    return w >= 0
      ? 'rgba(46,125,50,' + Math.min(a, 0.8).toFixed(2) + ')'
      : 'rgba(198,40,40,' + Math.min(a, 0.8).toFixed(2) + ')';
  }}
  function edgeWidth(w) {{
    if (maxAbsW === 0) return 1;
    const t = Math.abs(w) / maxAbsW;
    return {_MIN_EDGE_WIDTH} + t * {_MAX_EDGE_WIDTH - _MIN_EDGE_WIDTH};
  }}

  function clearEdges() {{
    while (edgesG.firstChild) edgesG.removeChild(edgesG.firstChild);
  }}

  function showEdgesFor(idx) {{
    clearEdges();
    const eIndices = nodeEdgeMap[idx] || [];
    const connectedNodes = new Set();
    connectedNodes.add(idx);
    eIndices.forEach(ei => {{
      const [src, tgt, w] = edgeData[ei];
      const line = document.createElementNS(SVG_NS, 'line');
      line.setAttribute('x1', nodePos[src][0]);
      line.setAttribute('y1', nodePos[src][1]);
      line.setAttribute('x2', nodePos[tgt][0]);
      line.setAttribute('y2', nodePos[tgt][1]);
      line.setAttribute('stroke', edgeColor(w));
      line.setAttribute('stroke-width', edgeWidth(w).toFixed(2));
      line.style.pointerEvents = 'none';
      edgesG.appendChild(line);
      connectedNodes.add(src);
      connectedNodes.add(tgt);
    }});
    // Dim unconnected nodes
    allNodes.forEach(g => {{
      const nIdx = parseInt(g.dataset.idx);
      if (connectedNodes.has(nIdx)) {{
        g.classList.remove('dim');
      }} else {{
        g.classList.add('dim');
      }}
    }});
    // Pin the node's label + connections in the info panel
    let html = tooltips[idx];
    const conns = eIndices.map(ei => {{
      const [src, tgt, w] = edgeData[ei];
      const isOut = src === idx;
      return {{ other: isOut ? tgt : src, w: w, dir: isOut ? '→ to' : '← from' }};
    }});
    conns.sort((a, b) => Math.abs(b.w) - Math.abs(a.w));
    if (conns.length) {{
      html += '<hr style="border:none;border-top:1px solid #ccc;margin:8px 0 6px;">';
      html += '<b>Connections (' + conns.length + ')</b>';
      conns.slice(0, 20).forEach(c => {{
        const col = c.w >= 0 ? '#2e7d32' : '#c62828';
        html += '<div style="white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
              + c.dir + ' ' + nodeLabels[c.other]
              + ' <span style="color:' + col + ';font-weight:bold;">' + c.w.toFixed(3) + '</span></div>';
      }});
      if (conns.length > 20) html += '<div style="color:#888;">… ' + (conns.length - 20) + ' more</div>';
    }}
    info.innerHTML = html;
    info.style.display = 'block';
  }}

  function deselect() {{
    selectedIdx = null;
    clearEdges();
    info.style.display = 'none';
    allNodes.forEach(g => g.classList.remove('dim'));
  }}

  // Click to select/deselect
  allNodes.forEach(g => {{
    const idx = parseInt(g.dataset.idx);
    g.addEventListener('click', e => {{
      e.stopPropagation();
      if (selectedIdx === idx) {{
        deselect();
      }} else {{
        selectedIdx = idx;
        showEdgesFor(idx);
      }}
    }});
    // Tooltip on hover
    g.addEventListener('mouseenter', e => {{
      tip.innerHTML = tooltips[idx];
      tip.style.display = 'block';
    }});
    g.addEventListener('mousemove', e => {{
      tip.style.left = (e.clientX + 14) + 'px';
      tip.style.top = (e.clientY + 14) + 'px';
    }});
    g.addEventListener('mouseleave', () => {{
      tip.style.display = 'none';
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
    html_str = render_graph_html_str(graph_dict, title=title, width=width, height=height)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html_str, encoding="utf-8")
    return output_path


def render_graph_html_str(
    graph_dict: dict,
    *,
    title: str | None = None,
    width: int = 1200,
    height: int = 800,
) -> str:
    """Return the self-contained HTML for *graph_dict* as a string (see
    :func:`render_graph_html`, which writes it to disk)."""
    nodes = graph_dict.get("nodes", [])
    edges = graph_dict.get("edges", [])
    tokens: list[str] | None = graph_dict.get("tokens")
    logit_token_strs: dict[str, str] | None = graph_dict.get("logit_token_strs")

    if title is None:
        title = graph_dict.get("prompt", "Attribution Graph")

    layout, actual_width = _compute_layout(nodes, width, height)
    return _render_html(nodes, edges, layout, tokens, logit_token_strs, title, actual_width, height)


def render_suite_html(
    entries: list[dict],
    output_path: str | Path,
    *,
    title: str = "Attribution graph suite",
) -> Path:
    """Render several graphs into one file with a dropdown to switch between them.

    Each entry is ``{"label": str, "summary": str (HTML), "graph_html": str}``
    where ``graph_html`` is a full document from :func:`render_graph_html_str`.
    Each graph is embedded in an isolated ``<iframe srcdoc=...>`` so the per-graph
    scripts/ids never collide, and the whole thing works offline (``file://``).
    """
    output_path = Path(output_path)
    if not entries:
        raise ValueError("render_suite_html requires at least one entry")

    options = "".join(
        f'<option value="{i}">{html.escape(e["label"])}</option>' for i, e in enumerate(entries)
    )
    summaries = "".join(
        f'<div class="gsum" data-idx="{i}" style="display:{"block" if i == 0 else "none"}">'
        f"{e.get('summary', '')}</div>"
        for i, e in enumerate(entries)
    )
    frames = "".join(
        f'<iframe class="gframe" data-idx="{i}" '
        f'style="display:{"block" if i == 0 else "none"}" '
        f'srcdoc="{html.escape(e["graph_html"], quote=True)}"></iframe>'
        for i, e in enumerate(entries)
    )
    title_esc = html.escape(title)
    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title_esc}</title>
<style>
  body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #fafafa; }}
  header {{ padding: 10px 16px; border-bottom: 1px solid #ddd; background: #fff; position: sticky; top: 0; z-index: 10; }}
  h2 {{ display: inline-block; margin: 0 12px 0 0; font-size: 16px; color: #333; }}
  select {{ font-size: 14px; padding: 3px 6px; }}
  .gsum {{ margin: 6px 0 0; font-size: 13px; color: #444; }}
  .gframe {{ width: 100%; height: 86vh; border: 0; }}
</style>
</head>
<body>
<header>
  <h2>{title_esc}</h2>
  <label>Example: <select id="pick">{options}</select></label>
  {summaries}
</header>
{frames}
<script>
(function() {{
  const pick = document.getElementById('pick');
  function show(idx) {{
    document.querySelectorAll('.gframe, .gsum').forEach(el => {{
      el.style.display = (el.dataset.idx === String(idx)) ? 'block' : 'none';
    }});
  }}
  pick.addEventListener('change', e => show(e.target.value));
}})();
</script>
</body>
</html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(doc, encoding="utf-8")
    return output_path
