"""Interactive attribution-graph explorer (self-contained HTML).

Renders one or more pruned :class:`AttributionGraph` dicts into a single static
HTML page that reproduces the components of Anthropic's circuit viewer:

* the **attribution graph** (position x layer), pan/zoom, click to inspect;
* a node **detail panel** — input features, output features, token predictions,
  and max-activating **activation examples** (token highlighting);
* **manual grouping**: shift-click nodes, name a group, and the **subgraph**
  collapses your groups into a supergraph (with summed edges).  Groups can be
  renamed / removed and **exported** to JSON.  Grouping is kept per graph.
* an **example dropdown** to switch between graphs.

Everything is embedded; no server or model is needed.  Each graph dict should
carry per-feature ``label`` with ``top_logits`` / ``bottom_logits`` and (optional)
``examples`` (from :func:`llm_circuits.transcoders.feature_labels.load_feature_examples`).
"""

from __future__ import annotations

import html
import json
from pathlib import Path

from llm_circuits.circuits.visualization import _compute_layout, _make_visible


def _raw_short_label(nd: dict, tokens, logit_token_strs) -> str:
    """A compact, *unescaped* one-line label (the JS escapes via textContent)."""
    t = nd["node_type"]
    layer, pos = nd["layer"], nd["position"]
    if t == "feature":
        top = ((nd.get("label") or {}).get("top_logits")) or []
        name = top[0] if top else f"f{nd['feature_idx']}"
        return f"L{layer} f{nd['feature_idx']} ({name})"
    if t == "embedding":
        tok = tokens[pos] if (tokens and 0 <= pos < len(tokens)) else f"p{pos}"
        return f'emb "{_make_visible(tok)}" @{pos}'
    if t == "error":
        return f"err L{layer} @{pos}"
    if t == "logit":
        tid = nd.get("token_id")
        s = logit_token_strs.get(str(tid)) if (logit_token_strs and tid is not None) else None
        return f'logit "{s}"' if s else f"logit {tid}"
    return t


def _derive_label(gd: dict, gi: int) -> str:
    prompt = gd.get("prompt") or f"example {gi + 1}"
    ans = gd.get("answer_token", "")
    core = prompt.split("?")[0].replace("What is", "").strip()
    label = f"{core}={ans}".strip()
    return label if label not in ("", "=") else (prompt[:24] or f"example {gi + 1}")


def _graph_payload(gd: dict, label: str, width: int, height: int) -> dict:
    nodes = gd.get("nodes", [])
    edges = gd.get("edges", [])
    tokens = gd.get("tokens")
    lts = gd.get("logit_token_strs")
    layout, actual_w = _compute_layout(nodes, width, height)
    payload_nodes = []
    for nd in nodes:
        lab = nd.get("label") or {}
        payload_nodes.append(
            {
                "t": nd["node_type"],
                "layer": nd["layer"],
                "pos": nd["position"],
                "f": nd.get("feature_idx"),
                "tok": nd.get("token_id"),
                "act": nd.get("activation", 0.0),
                "short": _raw_short_label(nd, tokens, lts),
                "top": (lab.get("top_logits") or [])[:10],
                "bot": (lab.get("bottom_logits") or [])[:10],
                "ex": lab.get("examples") or [],
                "freq": lab.get("activation_frequency"),
                "amin": lab.get("act_min"),
                "amax": lab.get("act_max"),
                "hist": lab.get("histogram") or [],
                "qv": lab.get("quantile_values") or [],
            }
        )
    return {
        "label": label,
        "nodes": payload_nodes,
        "edges": [[e["source"], e["target"], round(e["weight"], 5)] for e in edges],
        "pos": [[round(layout[i]["x"], 1), round(layout[i]["y"], 1)] for i in range(len(nodes))],
        "w": actual_w,
        "h": height,
        "yticks": _y_ticks(nodes, layout),
        "xticks": _x_ticks(nodes, layout, tokens),
    }


def _y_ticks(nodes: list[dict], layout: list[dict]) -> list[dict]:
    """One label per layer (subset): ``emb`` (input), ``output`` (logits), then L0/L5/...."""
    layer_y: dict[int, float] = {}
    logit_layers = {nd["layer"] for nd in nodes if nd["node_type"] == "logit"}
    other_layers = {nd["layer"] for nd in nodes if nd["node_type"] != "logit"}
    for i, nd in enumerate(nodes):
        layer_y.setdefault(nd["layer"], layout[i]["y"])
    ticks = []
    for layer in sorted(layer_y):
        if layer == -1:
            lab = "emb"
        elif layer in logit_layers and layer not in other_layers:
            lab = "output"
        elif layer % 5 == 0:
            lab = f"L{layer}"
        else:
            continue
        ticks.append({"y": round(layer_y[layer], 1), "label": lab})
    return ticks


def _x_ticks(nodes: list[dict], layout: list[dict], tokens) -> list[dict]:
    """One label per sequence position present in the graph: the input token string."""
    pos_x: dict[int, list[float]] = {}
    for i, nd in enumerate(nodes):
        x = layout[i]["x"]
        rng = pos_x.setdefault(nd["position"], [x, x])
        rng[0], rng[1] = min(rng[0], x), max(rng[1], x)
    ticks = []
    for pos in sorted(pos_x):
        mn, mx = pos_x[pos]
        tok = tokens[pos] if (tokens and 0 <= pos < len(tokens)) else f"p{pos}"
        ticks.append({"x": round((mn + mx) / 2, 1), "label": _make_visible(tok)[:14]})
    return ticks


def render_graph_explorer_html(
    graphs: dict | list[dict],
    output_path: str | Path,
    *,
    labels: list[str] | None = None,
    title: str = "Attribution graph explorer",
    width: int = 1100,
    height: int = 620,
) -> Path:
    """Render one or more graph dicts into a self-contained interactive explorer.

    *graphs* may be a single graph dict or a list of them; a dropdown switches
    between them and each keeps its own manual grouping.  *labels* optionally
    overrides the per-graph dropdown labels (else derived from ``prompt``).
    """
    if isinstance(graphs, dict):
        graphs = [graphs]
    examples = [
        _graph_payload(gd, (labels[i] if labels else None) or _derive_label(gd, i), width, height)
        for i, gd in enumerate(graphs)
    ]
    payload = {"title": title, "examples": examples}
    data_js = json.dumps(payload).replace("</", "<\\/")  # safe to embed in <script>
    doc = _TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__DATA__", data_js)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(doc, encoding="utf-8")
    return output_path


# NOTE: plain string (NOT an f-string) so the embedded JS can use { } and ${ }
# freely; only __TITLE__ / __DATA__ are substituted.
_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
         background:#fafafa; color:#222; font-size:13px; }
  header { padding:8px 14px; border-bottom:1px solid #ddd; background:#fff; }
  h2 { display:inline-block; margin:0 12px 0 0; font-size:15px; }
  #toolbar { font-size:12px; color:#555; }
  #toolbar input { font-size:12px; padding:2px 6px; width:140px; }
  #pick { font-size:13px; padding:2px 6px; margin-right:10px; }
  button { font-size:12px; padding:2px 9px; cursor:pointer; }
  #main { display:flex; height:calc(100vh - 46px); }
  #left { flex:1; display:flex; flex-direction:column; min-width:0; border-right:1px solid #ddd; }
  #right { width:380px; overflow:auto; padding:8px 12px; }
  .ptitle { font-size:10px; text-transform:uppercase; letter-spacing:.05em; color:#999; margin:4px 10px 0; }
  .legend { text-transform:none; letter-spacing:0; color:#888; margin-left:6px; }
  .legend svg { vertical-align:middle; margin:0 2px 0 9px; }
  #graphwrap { flex:1.6; overflow:hidden; }
  #subwrap { flex:1; overflow:hidden; border-top:1px solid #eee; }
  #g, #sg { display:block; background:#fff; width:100%; height:100%; cursor:grab; }
  #g { background:rgb(235,212,178); }
  #g:active, #sg:active { cursor:grabbing; }
  .node.dim { opacity:.15; }
  .frow { display:flex; justify-content:space-between; gap:8px; padding:1px 3px; white-space:nowrap; }
  .frow .nm { overflow:hidden; text-overflow:ellipsis; }
  .frow.nav { cursor:pointer; border-radius:3px; }
  .frow.nav:hover { background:#eef4ff; }
  .pos { color:#2e7d32; } .neg { color:#c62828; }
  .qgrp { margin:2px 0 8px; }
  .qname { font-size:10px; text-transform:uppercase; letter-spacing:.04em; color:#9a6a00; margin-top:6px; font-weight:600; }
  h3 { font-size:11px; text-transform:uppercase; letter-spacing:.04em; color:#666;
       margin:14px 0 4px; border-bottom:1px solid #eee; padding-bottom:3px; }
  .ex { font-family:ui-monospace,Menlo,monospace; font-size:11px; line-height:1.8; margin:3px 0; }
  .tk { padding:0 1px; border-radius:2px; }
  .chip { display:inline-block; padding:1px 5px; margin:1px; border-radius:3px; background:#eef; font-size:11px; }
  .chip.bot { background:#fee; }
  #groups .g { display:flex; align-items:center; gap:6px; border:1px solid #eee; border-radius:4px;
               padding:3px 6px; margin:3px 0; }
  #groups .g .nm { flex:1; }
  .sw { width:10px; height:10px; border-radius:50%; display:inline-block; }
  #hint { color:#999; font-size:11px; }
</style>
</head>
<body>
<header>
  <h2>__TITLE__</h2>
  <span id="toolbar">
    <label>example <select id="pick"></select></label>
    <span id="hint">click = inspect &nbsp;·&nbsp; shift-click = add to group</span> &nbsp;
    <input id="gname" type="text" placeholder="group name">
    <button id="mk">Group selected (<span id="seln">0</span>)</button>
    <button id="clr">Clear</button>
    <button id="exp">Export groups</button>
  </span>
</header>
<div id="main">
  <div id="left">
    <div class="ptitle">Attribution graph &mdash; position (x) &times; layer (y)
      <span class="legend">
        <svg width="13" height="13"><circle cx="6.5" cy="6.5" r="4.5" fill="none" stroke="#333"/></svg>feature
        <svg width="13" height="13"><rect x="3" y="3" width="7" height="7" fill="none" stroke="#333" transform="rotate(45 6.5 6.5)"/></svg>error
        <svg width="13" height="13"><rect x="2.5" y="2.5" width="8" height="8" fill="none" stroke="#333"/></svg>embed
        <svg width="13" height="13"><polygon points="6.5,2 11,11 2,11" fill="none" stroke="#333"/></svg>logit
        &middot; fill = group</span></div>
    <div id="graphwrap"><svg id="g"></svg></div>
    <div class="ptitle">Subgraph &mdash; your groups collapsed (drag a box to move &middot; click a member to inspect)</div>
    <div id="subwrap"><svg id="sg"></svg></div>
  </div>
  <div id="right">
    <div id="detail"></div>
    <h3>Groups</h3>
    <div id="groups"></div>
  </div>
</div>
<script>
const D = __DATA__;
const EX = D.examples;
const SVGNS = "http://www.w3.org/2000/svg";
// Node TYPE is encoded purely by SHAPE (no fill); fill is reserved for GROUP color.
const PAL = ["#8e24aa","#00897b","#f4511e","#3949ab","#c0ca33","#6d4c41","#00acc1","#d81b60"];
const PLACEHOLDER = "<em>Click a node to inspect its inputs, outputs, token predictions, and activation examples.</em>";

let cur = 0, N, E, POS, W, H, XT, YT, amin, amax, arange, MAXW;
const allGroups = EX.map(() => []);
let groups = allGroups[0];
let selected = null;
const selecting = new Set();
let nodeEls = [];
let subDrag = null;  // active subgraph supernode drag: {k, px, py, cx, cy}

function radius(a) { return 4 + 8 * ((Math.abs(a) - amin) / arange); }
function edgeColor(w) { const a = 0.15 + 0.6*(Math.abs(w)/MAXW);
  return w>=0 ? `rgba(46,125,50,${Math.min(a,.8)})` : `rgba(198,40,40,${Math.min(a,.8)})`; }
function edgeWidth(w) { return 0.4 + 3.2*(Math.abs(w)/MAXW); }
function esc(s){ const d=document.createElement("div"); d.textContent = s==null?"":String(s); return d.innerHTML; }

const g = document.getElementById("g"), sg = document.getElementById("sg");
const axesG = document.createElementNS(SVGNS,"g"); g.appendChild(axesG);  // behind edges/nodes
const edgesG = document.createElementNS(SVGNS,"g"); g.appendChild(edgesG);
const nodesG = document.createElementNS(SVGNS,"g"); g.appendChild(nodesG);

// Axis labels: layer number on the y-axis (emb at the bottom, output at the top),
// input token on the x-axis (rotated 45 deg). Drawn in graph coords (pan/zoom with the plot).
function drawAxes() {
  while (axesG.firstChild) axesG.removeChild(axesG.firstChild);
  const bottomY = (YT && YT.length) ? Math.max(...YT.map(t=>t.y)) : H;
  (YT||[]).forEach(t => {
    const ln = document.createElementNS(SVGNS,"line");
    ln.setAttribute("x1",0); ln.setAttribute("y1",t.y); ln.setAttribute("x2",W); ln.setAttribute("y2",t.y);
    ln.setAttribute("stroke","#eee"); ln.setAttribute("stroke-width","1"); axesG.appendChild(ln);
    const tx = document.createElementNS(SVGNS,"text");
    tx.setAttribute("x",4); tx.setAttribute("y",t.y-2); tx.setAttribute("font-size","11");
    tx.setAttribute("fill","#999"); tx.setAttribute("font-weight","600"); tx.textContent=t.label;
    axesG.appendChild(tx);
  });
  const ty = bottomY + 14;
  (XT||[]).forEach(t => {
    const tx = document.createElementNS(SVGNS,"text");
    tx.setAttribute("x",t.x); tx.setAttribute("y",ty); tx.setAttribute("font-size","11");
    tx.setAttribute("fill","#666"); tx.setAttribute("text-anchor","end");
    tx.setAttribute("transform",`rotate(-45 ${t.x} ${ty})`);
    tx.textContent=t.label; axesG.appendChild(tx);
  });
}

const tip = document.createElement("div");
tip.style.cssText = "display:none;position:fixed;padding:4px 8px;background:rgba(30,30,30,.92);color:#eee;border-radius:5px;font-size:12px;pointer-events:none;z-index:100;";
document.body.appendChild(tip);

// Distinct SHAPE per node type: feature=circle, error=diamond, embedding=square,
// logit=triangle. Fill defaults to white (hollow); group color is applied by repaintNodes.
function glyph(n, x, y, r) {
  let el;
  if (n.t === "feature") { el = document.createElementNS(SVGNS,"circle");
    el.setAttribute("cx",x); el.setAttribute("cy",y); el.setAttribute("r",r); }
  else if (n.t === "error") { el = document.createElementNS(SVGNS,"rect");
    const s=r*1.6; el.setAttribute("x",x-s/2); el.setAttribute("y",y-s/2);
    el.setAttribute("width",s); el.setAttribute("height",s); el.setAttribute("transform",`rotate(45 ${x} ${y})`); }
  else if (n.t === "logit") { el = document.createElementNS(SVGNS,"polygon");
    const s=r*1.15; el.setAttribute("points",`${x},${y-s} ${x-s},${y+s*0.85} ${x+s},${y+s*0.85}`); }
  else { el = document.createElementNS(SVGNS,"rect"); const s=r*1.7;  // embedding
    el.setAttribute("x",x-s/2); el.setAttribute("y",y-s/2); el.setAttribute("width",s); el.setAttribute("height",s); }
  el.setAttribute("class","glyph");
  el.setAttribute("fill","#fff"); el.setAttribute("stroke","#333"); el.setAttribute("stroke-width","1");
  return el;
}
// Fill encodes GROUP membership only (shape already encodes type); ungrouped = hollow (white).
function groupIdxOf(i){ for(let k=0;k<groups.length;k++) if(groups[k].members.includes(i)) return k; return -1; }
function nodeFill(i){ const k=groupIdxOf(i); return k>=0 ? groups[k].color : "#fff"; }

function clearEdges(){ while(edgesG.firstChild) edgesG.removeChild(edgesG.firstChild); }

function buildMain() {
  while (nodesG.firstChild) nodesG.removeChild(nodesG.firstChild);
  clearEdges(); drawAxes(); nodeEls = [];
  N.forEach((n,i) => {
    const [x,y] = POS[i], r = radius(n.act);
    const grp = document.createElementNS(SVGNS,"g");
    grp.setAttribute("class","node"); grp.dataset.idx = i;
    grp.appendChild(glyph(n, x, y, r));
    grp.addEventListener("click", ev => {
      ev.stopPropagation();
      if (ev.shiftKey) { selecting.has(i) ? selecting.delete(i) : selecting.add(i); repaintNodes(); }
      else selectNode(i);
    });
    grp.addEventListener("mouseenter", () => { tip.innerHTML = esc(n.short); tip.style.display="block"; });
    grp.addEventListener("mousemove", ev => { tip.style.left=(ev.clientX+12)+"px"; tip.style.top=(ev.clientY+12)+"px"; });
    grp.addEventListener("mouseleave", () => { tip.style.display="none"; });
    nodesG.appendChild(grp); nodeEls.push(grp);
  });
}

function showEdges(idx) {
  clearEdges();
  const conn = new Set([idx]);
  E.forEach(([s,t,w]) => {
    if (s!==idx && t!==idx) return;
    const ln = document.createElementNS(SVGNS,"line");
    ln.setAttribute("x1",POS[s][0]); ln.setAttribute("y1",POS[s][1]);
    ln.setAttribute("x2",POS[t][0]); ln.setAttribute("y2",POS[t][1]);
    ln.setAttribute("stroke",edgeColor(w)); ln.setAttribute("stroke-width",edgeWidth(w).toFixed(2));
    edgesG.appendChild(ln); conn.add(s); conn.add(t);
  });
  nodeEls.forEach((el,i) => el.classList.toggle("dim", !conn.has(i)));
  repaintNodes();
}

// Repaint node fills (group color) + selection/selecting rings.
function repaintNodes() {
  document.getElementById("seln").textContent = selecting.size;
  nodeEls.forEach((el,i) => {
    const shape = el.querySelector(".glyph"); if (shape) shape.setAttribute("fill", nodeFill(i));
    let ring = el.querySelector(".ring"), color = null;
    if (selecting.has(i)) color = "#ff9800";
    if (i === selected) color = "#e91e63";
    if (color) {
      if (!ring) { ring = document.createElementNS(SVGNS,"circle"); ring.setAttribute("class","ring");
        ring.setAttribute("cx",POS[i][0]); ring.setAttribute("cy",POS[i][1]); ring.setAttribute("r",radius(N[i].act)+3);
        ring.setAttribute("fill","none"); ring.setAttribute("stroke-width","2.4"); el.appendChild(ring); }
      ring.setAttribute("stroke",color);
    } else if (ring) ring.remove();
  });
}

function featRows(idx, incoming) {
  const rows = E.filter(e => incoming ? e[1]===idx : e[0]===idx)
                .map(e => ({other: incoming ? e[0] : e[1], w: e[2]}))
                .sort((a,b)=>Math.abs(b.w)-Math.abs(a.w)).slice(0,15);
  if (!rows.length) return "<div style='color:#999'>none</div>";
  return rows.map(r => `<div class="frow nav" data-idx="${r.other}" title="click to select">`
    + `<span class="nm">${esc(N[r.other].short)}</span>`
    + `<span class="${r.w>=0?'pos':'neg'}">${r.w>=0?'+':''}${r.w.toFixed(3)}</span></div>`).join("");
}
function chips(arr, bot) {
  if (!arr || !arr.length) return "<span style='color:#999'>n/a</span>";
  return arr.map(t => `<span class="chip${bot?' bot':''}">${esc(t)}</span>`).join("");
}
function exLine(ex, scale) {
  // Highlight by SIGNED activation: green = positive, red = negative; intensity = |value|/scale.
  const spans = ex.tokens.map((tk,j) => {
    const v = ex.acts[j]||0, mag = Math.abs(v)/scale;
    let bg = "";
    if (mag > 0.02) { const al = Math.min(0.15+0.85*mag,1).toFixed(2);
      bg = `background:rgba(${v>=0?'46,125,50':'198,40,40'},${al})`; }
    return `<span class="tk" style="${bg}">${esc(tk)}</span>`;
  }).join("");
  return `<div class="ex">${spans}</div>`;
}
function examplesHtml(n) {
  if (!n.ex || !n.ex.length) return "<div style='color:#999'>no activation examples</div>";
  // n.ex is quantile-grouped: [{quantile, items:[{tokens,acts}]}]; scale by max |act|.
  return n.ex.map(q => {
    const scale = Math.max(...q.items.flatMap(it => it.acts.map(a => Math.abs(a))), 1e-6);
    const lines = q.items.map(it => exLine(it, scale)).join("");
    return `<div class="qgrp"><div class="qname">${esc(q.quantile)}</div>${lines}</div>`;
  }).join("");
}
// Activation histogram on a LOG activation-value x-axis, using the real (non-uniform)
// bin edges in n.qv (quantile_values): bar width = log-spaced bin width, height = sqrt(count).
// Falls back to equal-width bars only if bin edges are unavailable.
function histHtml(n) {
  const h = n.hist; if (!h || !h.length) return "";
  const w = 248, ht = 40, mxH = Math.sqrt(Math.max(...h, 1));
  const bh = c => (Math.sqrt(Math.max(c,0)) / mxH) * ht;
  const qv = n.qv;
  if (!qv || qv.length !== h.length + 1) {  // no bin edges -> plain equal-width fallback
    const bw = w / h.length;
    const bars = h.map((c,j) =>
      `<rect x="${(j*bw).toFixed(2)}" y="${(ht-bh(c)).toFixed(2)}" width="${Math.max(bw-0.3,0.4).toFixed(2)}" height="${bh(c).toFixed(2)}" fill="#2196F3"/>`).join("");
    return `<svg width="${w}" height="${ht}" style="display:block">${bars}</svg>`;
  }
  const hi = qv[qv.length-1];
  let posMin = Infinity; for (const v of qv) if (v>0 && v<posMin) posMin = v;
  if (!isFinite(posMin)) posMin = Math.max(hi,1e-6)*1e-3;
  const hiL = Math.log10(Math.max(hi, posMin*1.0001));
  // cap the axis at ~4 decades so a tiny act_min can't blow up the lowest bin;
  // values below 10^loL clamp to the left edge ("near zero").
  const loL = Math.max(Math.log10(posMin) - 0.3, hiL - 4);
  const span = (hiL - loL) || 1;
  const X = v => ((Math.min(Math.max(v>0?Math.log10(v):loL, loL), hiL) - loL) / span) * w;
  const bars = h.map((c,i) => {
    const x0 = X(qv[i]);
    return `<rect x="${x0.toFixed(2)}" y="${(ht-bh(c)).toFixed(2)}" width="${Math.max(X(qv[i+1])-x0,0.5).toFixed(2)}" height="${bh(c).toFixed(2)}" fill="#2196F3"/>`;
  }).join("");
  let grid = "", labels = "";
  for (let d=Math.ceil(loL); d<=Math.floor(hiL); d++) {
    const x = X(Math.pow(10,d)).toFixed(1);
    grid += `<line x1="${x}" y1="0" x2="${x}" y2="${ht}" stroke="#cfd8dc" stroke-width="0.6"/>`;
    labels += `<text x="${x}" y="${ht+10}" font-size="9" fill="#999" text-anchor="middle">${Math.pow(10,d)}</text>`;
  }
  return `<svg width="${w}" height="${ht+13}" style="display:block">${grid}${bars}${labels}</svg>`
    + `<div style="color:#999;font-size:10px;text-align:center">activation value (log scale)</div>`;
}
function showDetail(idx) {
  const n = N[idx];
  let head;
  if (n.t === "feature") head = `<b>F${n.f}</b> &nbsp; L${n.layer} · pos ${n.pos} &nbsp; <span style="color:#888">${esc(n.top[0]||"")}</span>`;
  else if (n.t === "logit") head = `<b>logit</b> ${esc(n.short)} &nbsp;(act ${n.act.toFixed(3)})`;
  else if (n.t === "embedding") head = `<b>embedding</b> ${esc(n.short)}`;
  else head = `<b>error</b> L${n.layer} · pos ${n.pos}`;
  let h = `<div style="font-size:13px;margin-bottom:2px">${head}</div>`;
  if (n.t === "feature") {
    const stats = [];
    if (n.freq != null) stats.push(`activation freq <b>${(n.freq*100).toPrecision(3)}%</b>`);
    if (n.amax != null) stats.push(`act range <b>${(+n.amin).toFixed(2)}&ndash;${(+n.amax).toFixed(2)}</b>`);
    stats.push(`peak act <b>${n.act.toFixed(3)}</b> (this token)`);
    h += `<div style="color:#555;font-size:11px;margin:2px 0">${stats.join(" &nbsp;·&nbsp; ")}</div>`;
  }
  h += `<h3>Input features (&rarr; this node)</h3>${featRows(idx,true)}`;
  h += `<h3>Output features (this node &rarr;)</h3>${featRows(idx,false)}`;
  if (n.t === "feature") {
    h += `<h3>Token predictions</h3><div><b style="font-size:11px">top</b> ${chips(n.top,false)}</div>`;
    h += `<div style="margin-top:3px"><b style="font-size:11px">bottom</b> ${chips(n.bot,true)}</div>`;
    const hh = histHtml(n);
    if (hh) h += `<h3>Activation distribution</h3>${hh}`;
    h += `<h3>Activation examples</h3>${examplesHtml(n)}`;
  }
  document.getElementById("detail").innerHTML = h;
  repaintNodes();
}

// Subgraph: ONLY the user's groups, each a box containing its member glyphs.
// Ungrouped nodes are hidden. Members are clickable (select + inspect in main).
// Group<->group edges are summed. fit=true refits the viewBox (structural change).
function drawSub(fit) {
  while (sg.firstChild) sg.removeChild(sg.firstChild);
  if (!groups.length) {
    const t=document.createElementNS(SVGNS,"text"); t.setAttribute("x",20); t.setAttribute("y",30);
    t.setAttribute("fill","#bbb"); t.setAttribute("font-size","12");
    t.textContent="Shift-click nodes above and 'Group selected' to build the subgraph.";
    sg.appendChild(t); if (fit) resetSub(600,200,0,0); return;
  }
  const eG = document.createElementNS(SVGNS,"g"), nG = document.createElementNS(SVGNS,"g");
  sg.appendChild(eG); sg.appendChild(nG);

  // Each group box centered on its manual position (if dragged) else the average
  // position of its members (which preserves layer/x).
  const GAP=22, PAD=10, RG=6, BH=26;
  const boxes = groups.map(gr => {
    let cx, cy;
    if (gr.pos) { cx=gr.pos.x; cy=gr.pos.y; }
    else { let xs=0, ys=0; gr.members.forEach(i => { xs+=POS[i][0]; ys+=POS[i][1]; });
      cx=xs/gr.members.length; cy=ys/gr.members.length; }
    const bw=Math.max(gr.members.length*GAP + 2*PAD, 46);
    return {cx, cy, bw, left:cx-bw/2};
  });

  // Group<->group aggregated edges (skip edges that touch ungrouped nodes).
  const agg = {};
  E.forEach(([s,t,w]) => { const a=groupIdxOf(s), b=groupIdxOf(t);
    if (a<0 || b<0 || a===b) return; const k=a+"|"+b; agg[k]=(agg[k]||0)+w; });
  let mw=1; for(const k in agg) mw=Math.max(mw,Math.abs(agg[k]));
  for (const k in agg){ const [a,b]=k.split("|").map(Number), w=agg[k];
    const ln=document.createElementNS(SVGNS,"line");
    ln.setAttribute("x1",boxes[a].cx); ln.setAttribute("y1",boxes[a].cy);
    ln.setAttribute("x2",boxes[b].cx); ln.setAttribute("y2",boxes[b].cy);
    const al=0.2+0.6*(Math.abs(w)/mw);
    ln.setAttribute("stroke", w>=0?`rgba(46,125,50,${al.toFixed(2)})`:`rgba(198,40,40,${al.toFixed(2)})`);
    ln.setAttribute("stroke-width",(0.6+3.4*(Math.abs(w)/mw)).toFixed(2)); eG.appendChild(ln); }

  // Draw boxes + member glyphs (clickable) + labels. Box/label = drag handle (move the
  // supernode); member glyphs stay clickable (select). See subDrag wiring below.
  groups.forEach((gr,k) => {
    const b=boxes[k];
    const node=document.createElementNS(SVGNS,"g"); node.setAttribute("class","supernode");
    const box=document.createElementNS(SVGNS,"rect");
    box.setAttribute("x",b.left); box.setAttribute("y",b.cy-BH/2);
    box.setAttribute("width",b.bw); box.setAttribute("height",BH); box.setAttribute("rx",6);
    box.setAttribute("fill","#fff"); box.setAttribute("stroke",gr.color); box.setAttribute("stroke-width","2.2");
    box.style.cursor="move"; node.appendChild(box);
    gr.members.forEach((i,j) => {
      const gx=b.left+PAD+(j+0.5)*GAP;
      const el=glyph(N[i], gx, b.cy, RG);
      el.setAttribute("fill", gr.color); el.style.cursor="pointer";
      if (i===selected) { el.setAttribute("stroke","#e91e63"); el.setAttribute("stroke-width","2.4"); }
      el.addEventListener("click", ev => { ev.stopPropagation(); selectNode(i); });
      el.addEventListener("mouseenter", () => { tip.innerHTML=esc(N[i].short); tip.style.display="block"; });
      el.addEventListener("mousemove", ev => { tip.style.left=(ev.clientX+12)+"px"; tip.style.top=(ev.clientY+12)+"px"; });
      el.addEventListener("mouseleave", () => { tip.style.display="none"; });
      node.appendChild(el);
    });
    const tx=document.createElementNS(SVGNS,"text"); tx.setAttribute("x",b.cx); tx.setAttribute("y",b.cy+BH/2+13);
    tx.setAttribute("text-anchor","middle"); tx.setAttribute("font-size","11"); tx.setAttribute("fill",gr.color);
    tx.setAttribute("font-weight","600"); tx.style.cursor="move"; tx.textContent=gr.name.slice(0,22);
    node.appendChild(tx);
    // Drag the box/label (not member glyphs) to reposition this supernode.
    node.addEventListener("mousedown", ev => {
      if (ev.target.classList && ev.target.classList.contains("glyph")) return;  // member -> click/select
      ev.stopPropagation();
      subDrag = {k, px:ev.clientX, py:ev.clientY, cx:b.cx, cy:b.cy};
    });
    nG.appendChild(node);
  });

  if (fit) {
    let x0=1e9,y0=1e9,x1=-1e9,y1=-1e9;
    boxes.forEach(b => { x0=Math.min(x0,b.left); x1=Math.max(x1,b.left+b.bw);
      y0=Math.min(y0,b.cy-BH/2); y1=Math.max(y1,b.cy+BH/2+18); });
    const m=40; resetSub((x1-x0)+2*m, (y1-y0)+2*m, x0-m, y0-m);
  }
}

function makeGroup() {
  if (!selecting.size) return;
  const name = (document.getElementById("gname").value || ("group "+(groups.length+1))).trim();
  groups.push({name, members:[...selecting], color: PAL[groups.length % PAL.length]});
  selecting.clear(); document.getElementById("gname").value="";
  renderGroups(); drawSub(true); repaintNodes();
}
function renderGroups() {
  const box = document.getElementById("groups");
  if (!groups.length) { box.innerHTML = "<div style='color:#999'>shift-click nodes, then 'Group selected'.</div>"; return; }
  box.innerHTML = "";
  groups.forEach((gr,k) => {
    const row = document.createElement("div"); row.className="g";
    row.innerHTML = `<span class="sw" style="background:${gr.color}"></span>`
      + `<input class="nm" value="${esc(gr.name)}" style="font-size:12px">`
      + `<span style="color:#888">${gr.members.length}</span> <button>x</button>`;
    row.querySelector("input").addEventListener("change", e => { gr.name=e.target.value; drawSub(false); });
    row.querySelector("button").addEventListener("click", () => { groups.splice(k,1); renderGroups(); drawSub(true); repaintNodes(); });
    box.appendChild(row);
  });
}
function exportGroups() {
  const out = {example: EX[cur].label, groups: groups.map(gr => ({name: gr.name,
    nodes: gr.members.map(i => ({label:N[i].short, layer:N[i].layer, position:N[i].pos, feature_idx:N[i].f}))}))};
  const blob = new Blob([JSON.stringify(out,null,2)], {type:"application/json"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob);
  a.download = "groups_" + EX[cur].label.replace(/[^A-Za-z0-9]+/g,"_") + ".json"; a.click();
}

function panzoom(svg) {
  let vb={x:0,y:0,w:1,h:1}, pan=false, sx=0, sy=0;
  function set(){ svg.setAttribute("viewBox",`${vb.x} ${vb.y} ${vb.w} ${vb.h}`); }
  svg.addEventListener("wheel", e => { e.preventDefault(); const k=e.deltaY>0?1.1:0.9;
    const r=svg.getBoundingClientRect(); const mx=(e.clientX-r.left)/r.width, my=(e.clientY-r.top)/r.height;
    const nw=vb.w*k, nh=vb.h*k; vb.x+=(vb.w-nw)*mx; vb.y+=(vb.h-nh)*my; vb.w=nw; vb.h=nh; set(); }, {passive:false});
  svg.addEventListener("mousedown", e => { if(e.target===svg || e.target.tagName==="line"){ pan=true; sx=e.clientX; sy=e.clientY; } });
  window.addEventListener("mousemove", e => { if(!pan) return; const r=svg.getBoundingClientRect();
    vb.x-=(e.clientX-sx)*(vb.w/r.width); vb.y-=(e.clientY-sy)*(vb.h/r.height); sx=e.clientX; sy=e.clientY; set(); });
  window.addEventListener("mouseup", () => pan=false);
  return function reset(w,h,x,y){ vb={x:x||0,y:y||0,w:w,h:h}; set(); };
}
const resetMain = panzoom(g), resetSub = panzoom(sg);

// --- Subgraph supernode dragging ---------------------------------------------
// SVG units per screen pixel for the sub svg (xMidYMid meet => uniform scale).
function subUnitsPerPx(){ const v=sg.getAttribute("viewBox"); if(!v) return 1;
  const p=v.trim().split(/ +/).map(Number), r=sg.getBoundingClientRect();
  return Math.max(p[2]/(r.width||1), p[3]/(r.height||1)); }
window.addEventListener("mousemove", e => {
  if (!subDrag) return;
  const s=subUnitsPerPx();
  groups[subDrag.k].pos = {x: subDrag.cx + (e.clientX-subDrag.px)*s, y: subDrag.cy + (e.clientY-subDrag.py)*s};
  drawSub(false);  // redraw at the new position (no refit, so the user's zoom is kept)
});
window.addEventListener("mouseup", () => { subDrag = null; });

// Select a node from anywhere (main graph, subgraph member, or a detail row).
function selectNode(i){ selected=i; showDetail(i); showEdges(i); drawSub(false); }
g.addEventListener("click", () => { selected=null; clearEdges(); nodeEls.forEach(el=>el.classList.remove("dim")); repaintNodes(); drawSub(false); });
// Click an input/output feature row to navigate to that node.
document.getElementById("detail").addEventListener("click", ev => {
  const row = ev.target.closest(".frow.nav"); if (row) selectNode(+row.dataset.idx);
});

function loadExample(i) {
  cur = i; const ex = EX[i];
  N = ex.nodes; E = ex.edges; POS = ex.pos; W = ex.w; H = ex.h;
  XT = ex.xticks; YT = ex.yticks;
  groups = allGroups[i]; selected = null; selecting.clear();
  const aa = N.map(n => Math.abs(n.act||0));
  amin = Math.min(...aa); amax = Math.max(...aa); arange = (amax - amin) || 1;
  MAXW = 1; for (const e of E) MAXW = Math.max(MAXW, Math.abs(e[2]));
  // extra bottom room (rotated x-labels) + small left pad so y-labels aren't clipped
  buildMain(); repaintNodes(); resetMain(W+12, H+80, -12, 0);
  document.getElementById("detail").innerHTML = PLACEHOLDER;
  renderGroups(); drawSub(true);
}

const pick = document.getElementById("pick");
EX.forEach((ex,i) => { const o=document.createElement("option"); o.value=i; o.textContent=ex.label; pick.appendChild(o); });
pick.addEventListener("change", e => loadExample(+e.target.value));
document.getElementById("mk").addEventListener("click", makeGroup);
document.getElementById("clr").addEventListener("click", () => { selecting.clear(); repaintNodes(); });
document.getElementById("exp").addEventListener("click", exportGroups);

loadExample(0);
</script>
</body>
</html>"""
