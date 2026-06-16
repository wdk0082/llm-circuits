"""Interactive attribution-graph explorer (self-contained HTML).

Renders a pruned :class:`AttributionGraph` (as a JSON dict) into a single static
HTML page that reproduces the components of Anthropic's circuit viewer:

* the **attribution graph** (position x layer), pan/zoom, click to inspect;
* a node **detail panel** — input features, output features, token predictions,
  and max-activating **activation examples** (token highlighting);
* **manual grouping**: shift-click nodes, name a group, and the **subgraph**
  collapses your groups into a supergraph (with summed edges).  Groups can be
  renamed / removed and **exported** to JSON.

Everything is embedded; no server or model is needed.  The graph dict should
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


def render_graph_explorer_html(
    graph_dict: dict,
    output_path: str | Path,
    *,
    title: str | None = None,
    width: int = 1100,
    height: int = 620,
) -> Path:
    """Render *graph_dict* into a self-contained interactive explorer HTML file."""
    nodes = graph_dict.get("nodes", [])
    edges = graph_dict.get("edges", [])
    tokens = graph_dict.get("tokens")
    lts = graph_dict.get("logit_token_strs")
    if title is None:
        title = graph_dict.get("prompt", "Attribution graph explorer")

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
                "top": (lab.get("top_logits") or [])[:8],
                "bot": (lab.get("bottom_logits") or [])[:6],
                "ex": lab.get("examples") or [],
            }
        )

    payload = {
        "title": title,
        "nodes": payload_nodes,
        "edges": [[e["source"], e["target"], round(e["weight"], 5)] for e in edges],
        "pos": [[round(layout[i]["x"], 1), round(layout[i]["y"], 1)] for i in range(len(nodes))],
        "w": actual_w,
        "h": height,
    }
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
  h2 { display:inline-block; margin:0 14px 0 0; font-size:15px; }
  #toolbar { font-size:12px; color:#555; }
  #toolbar input { font-size:12px; padding:2px 6px; width:140px; }
  button { font-size:12px; padding:2px 9px; cursor:pointer; }
  #main { display:flex; height:calc(100vh - 46px); }
  #left { flex:1; display:flex; flex-direction:column; min-width:0; border-right:1px solid #ddd; }
  #right { width:380px; overflow:auto; padding:8px 12px; }
  .ptitle { font-size:10px; text-transform:uppercase; letter-spacing:.05em; color:#999; margin:4px 10px 0; }
  #graphwrap { flex:1.6; overflow:hidden; }
  #subwrap { flex:1; overflow:hidden; border-top:1px solid #eee; }
  svg { display:block; background:#fff; width:100%; height:100%; cursor:grab; }
  svg:active { cursor:grabbing; }
  .node.dim { opacity:.15; }
  .frow { display:flex; justify-content:space-between; gap:8px; padding:1px 0; white-space:nowrap; }
  .frow .nm { overflow:hidden; text-overflow:ellipsis; }
  .pos { color:#2e7d32; } .neg { color:#c62828; }
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
    <span id="hint">click = inspect &nbsp;·&nbsp; shift-click = add to group</span> &nbsp;
    <input id="gname" type="text" placeholder="group name">
    <button id="mk">Group selected (<span id="seln">0</span>)</button>
    <button id="clr">Clear</button>
    <button id="exp">Export groups</button>
  </span>
</header>
<div id="main">
  <div id="left">
    <div class="ptitle">Attribution graph &mdash; position (x) &times; layer (y)</div>
    <div id="graphwrap"><svg id="g"></svg></div>
    <div class="ptitle">Subgraph &mdash; your groups collapsed</div>
    <div id="subwrap"><svg id="sg"></svg></div>
  </div>
  <div id="right">
    <div id="detail"><em>Click a node to inspect its inputs, outputs, token predictions,
      and activation examples.</em></div>
    <h3>Groups</h3>
    <div id="groups"></div>
  </div>
</div>
<script>
const D = __DATA__;
const N = D.nodes, E = D.edges, POS = D.pos;
const SVGNS = "http://www.w3.org/2000/svg";
const COL = {embedding:"#4CAF50", feature:"#2196F3", error:"#9E9E9E", logit:"#FF9800"};
const PAL = ["#8e24aa","#00897b","#f4511e","#3949ab","#c0ca33","#6d4c41","#00acc1","#d81b60"];

let selected = null;
const selecting = new Set();
let groups = [];   // {name, members:[idx], color}

// activation range for node sizing
const acts = N.map(n => Math.abs(n.act || 0));
const amin = Math.min(...acts), amax = Math.max(...acts), arange = (amax - amin) || 1;
function radius(a) { return 4 + 8 * ((Math.abs(a) - amin) / arange); }
function maxAbsW() { let m=0; for (const e of E) m = Math.max(m, Math.abs(e[2])); return m || 1; }
const MAXW = maxAbsW();
function edgeColor(w) { const a = 0.15 + 0.6*(Math.abs(w)/MAXW);
  return w>=0 ? `rgba(46,125,50,${Math.min(a,.8)})` : `rgba(198,40,40,${Math.min(a,.8)})`; }
function edgeWidth(w) { return 0.4 + 3.2*(Math.abs(w)/MAXW); }
function esc(s){ const d=document.createElement("div"); d.textContent = s==null?"":String(s); return d.innerHTML; }

// ---------- main graph ----------
const g = document.getElementById("g");
g.setAttribute("viewBox", `0 0 ${D.w} ${D.h}`);
const edgesG = document.createElementNS(SVGNS,"g"); g.appendChild(edgesG);
const nodesG = document.createElementNS(SVGNS,"g"); g.appendChild(nodesG);
const nodeEls = [];

function glyph(n, x, y, r) {
  const c = COL[n.t] || "#888";
  let el;
  if (n.t === "feature") { el = document.createElementNS(SVGNS,"circle");
    el.setAttribute("cx",x); el.setAttribute("cy",y); el.setAttribute("r",r); }
  else if (n.t === "error") { el = document.createElementNS(SVGNS,"rect");
    const s=r*1.6; el.setAttribute("x",x-s/2); el.setAttribute("y",y-s/2);
    el.setAttribute("width",s); el.setAttribute("height",s);
    el.setAttribute("transform",`rotate(45 ${x} ${y})`); }
  else { el = document.createElementNS(SVGNS,"rect"); const s=r*1.8;
    el.setAttribute("x",x-s/2); el.setAttribute("y",y-s/2);
    el.setAttribute("width",s); el.setAttribute("height",s); }
  el.setAttribute("fill",c); el.setAttribute("stroke","#333"); el.setAttribute("stroke-width","1");
  return el;
}

N.forEach((n,i) => {
  const [x,y] = POS[i], r = radius(n.act);
  const grp = document.createElementNS(SVGNS,"g");
  grp.setAttribute("class","node"); grp.dataset.idx = i;
  const el = glyph(n, x, y, r); grp.appendChild(el);
  grp.addEventListener("click", ev => {
    ev.stopPropagation();
    if (ev.shiftKey) { selecting.has(i) ? selecting.delete(i) : selecting.add(i); paintSelecting(); }
    else { selected = i; showDetail(i); showEdges(i); }
  });
  grp.addEventListener("mouseenter", () => { tip.innerHTML = esc(n.short); tip.style.display="block"; });
  grp.addEventListener("mousemove", ev => { tip.style.left=(ev.clientX+12)+"px"; tip.style.top=(ev.clientY+12)+"px"; });
  grp.addEventListener("mouseleave", () => { tip.style.display="none"; });
  nodesG.appendChild(grp); nodeEls.push(grp);
});

const tip = document.createElement("div");
tip.style.cssText = "display:none;position:fixed;padding:4px 8px;background:rgba(30,30,30,.92);color:#eee;border-radius:5px;font-size:12px;pointer-events:none;z-index:100;";
document.body.appendChild(tip);

function clearEdges(){ while(edgesG.firstChild) edgesG.removeChild(edgesG.firstChild); }
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
  paintSelecting();
}

function paintSelecting() {
  document.getElementById("seln").textContent = selecting.size;
  nodeEls.forEach((el,i) => {
    let ring = el.querySelector(".ring");
    let color = null;
    if (selecting.has(i)) color = "#ff9800";
    else { for (let k=0;k<groups.length;k++) if (groups[k].members.includes(i)) { color = groups[k].color; break; } }
    if (i === selected) color = "#e91e63";
    if (color) {
      if (!ring) { ring = document.createElementNS(SVGNS,"circle"); ring.setAttribute("class","ring");
        ring.setAttribute("cx",POS[i][0]); ring.setAttribute("cy",POS[i][1]);
        ring.setAttribute("r",radius(N[i].act)+3); ring.setAttribute("fill","none"); ring.setAttribute("stroke-width","2.2");
        el.appendChild(ring); }
      ring.setAttribute("stroke",color);
    } else if (ring) ring.remove();
  });
}

// ---------- detail panel ----------
function featRows(idx, incoming) {
  const rows = E.filter(e => incoming ? e[1]===idx : e[0]===idx)
                .map(e => ({other: incoming ? e[0] : e[1], w: e[2]}))
                .sort((a,b)=>Math.abs(b.w)-Math.abs(a.w)).slice(0,15);
  if (!rows.length) return "<div style='color:#999'>none</div>";
  return rows.map(r => `<div class="frow"><span class="nm">${esc(N[r.other].short)}</span>`
    + `<span class="${r.w>=0?'pos':'neg'}">${r.w>=0?'+':''}${r.w.toFixed(3)}</span></div>`).join("");
}
function chips(arr, bot) {
  if (!arr || !arr.length) return "<span style='color:#999'>n/a</span>";
  return arr.map(t => `<span class="chip${bot?' bot':''}">${esc(t)}</span>`).join("");
}
function examplesHtml(n) {
  if (!n.ex || !n.ex.length) return "<div style='color:#999'>no activation examples</div>";
  return n.ex.map(ex => {
    const mx = Math.max(...ex.acts, 1e-6);
    const spans = ex.tokens.map((tk,j) => {
      const a = (ex.acts[j]||0)/mx;
      const bg = a>0.02 ? `background:rgba(255,140,0,${(0.15+0.85*a).toFixed(2)})` : "";
      return `<span class="tk" style="${bg}">${esc(tk)}</span>`;
    }).join("");
    return `<div class="ex">${spans}</div>`;
  }).join("");
}
function showDetail(idx) {
  const n = N[idx];
  let head;
  if (n.t === "feature") head = `<b>F${n.f}</b> &nbsp; L${n.layer} · pos ${n.pos} &nbsp; <span style="color:#888">${esc((n.top[0]||""))}</span>`;
  else if (n.t === "logit") head = `<b>logit</b> ${esc(n.short)} &nbsp;(act ${n.act.toFixed(3)})`;
  else if (n.t === "embedding") head = `<b>embedding</b> ${esc(n.short)}`;
  else head = `<b>error</b> L${n.layer} · pos ${n.pos}`;
  let h = `<div style="font-size:13px;margin-bottom:2px">${head}</div>`;
  h += `<h3>Input features (&rarr; this node)</h3>${featRows(idx,true)}`;
  h += `<h3>Output features (this node &rarr;)</h3>${featRows(idx,false)}`;
  if (n.t === "feature") {
    h += `<h3>Token predictions</h3><div><b style="font-size:11px">top</b> ${chips(n.top,false)}</div>`;
    h += `<div style="margin-top:3px"><b style="font-size:11px">bottom</b> ${chips(n.bot,true)}</div>`;
    h += `<h3>Activation examples</h3>${examplesHtml(n)}`;
  }
  document.getElementById("detail").innerHTML = h;
  paintSelecting();
}

// ---------- subgraph (collapse groups) ----------
const sg = document.getElementById("sg");
sg.setAttribute("viewBox", `0 0 ${D.w} ${D.h}`);
function superOf(idx){ for(let k=0;k<groups.length;k++) if(groups[k].members.includes(idx)) return "g"+k; return "n"+idx; }
function drawSub() {
  while (sg.firstChild) sg.removeChild(sg.firstChild);
  const eG = document.createElementNS(SVGNS,"g"), nG = document.createElementNS(SVGNS,"g");
  sg.appendChild(eG); sg.appendChild(nG);
  // super-node positions
  const sup = {};
  N.forEach((n,i) => { const s=superOf(i);
    if(!sup[s]) sup[s]={xs:0,ys:0,c:0};
    sup[s].xs+=POS[i][0]; sup[s].ys+=POS[i][1]; sup[s].c++; });
  for (const s in sup){ sup[s].x=sup[s].xs/sup[s].c; sup[s].y=sup[s].ys/sup[s].c; }
  // aggregate edges
  const agg = {};
  E.forEach(([s,t,w]) => { const a=superOf(s), b=superOf(t); if(a===b) return;
    agg[a+"|"+b] = (agg[a+"|"+b]||0) + w; });
  let mw=1; for(const k in agg) mw=Math.max(mw,Math.abs(agg[k]));
  for (const k in agg){ const [a,b]=k.split("|"); const w=agg[k];
    const ln=document.createElementNS(SVGNS,"line");
    ln.setAttribute("x1",sup[a].x); ln.setAttribute("y1",sup[a].y);
    ln.setAttribute("x2",sup[b].x); ln.setAttribute("y2",sup[b].y);
    const al=0.15+0.6*(Math.abs(w)/mw);
    ln.setAttribute("stroke", w>=0?`rgba(46,125,50,${al.toFixed(2)})`:`rgba(198,40,40,${al.toFixed(2)})`);
    ln.setAttribute("stroke-width",(0.5+3*(Math.abs(w)/mw)).toFixed(2)); eG.appendChild(ln); }
  // nodes
  for (const s in sup){
    if (s[0]==="g"){ const k=+s.slice(1), gr=groups[k];
      const box=document.createElementNS(SVGNS,"rect"); box.setAttribute("x",sup[s].x-32);
      box.setAttribute("y",sup[s].y-11); box.setAttribute("width",64); box.setAttribute("height",22);
      box.setAttribute("rx",5); box.setAttribute("fill",gr.color); box.setAttribute("opacity","0.85"); nG.appendChild(box);
      const tx=document.createElementNS(SVGNS,"text"); tx.setAttribute("x",sup[s].x); tx.setAttribute("y",sup[s].y+4);
      tx.setAttribute("text-anchor","middle"); tx.setAttribute("font-size","9"); tx.setAttribute("fill","#fff");
      tx.textContent = gr.name.slice(0,12); nG.appendChild(tx);
    } else { const i=+s.slice(1), n=N[i];
      nG.appendChild(glyph(n, sup[s].x, sup[s].y, radius(n.act))); }
  }
}

// ---------- groups ----------
function makeGroup() {
  if (!selecting.size) return;
  const name = (document.getElementById("gname").value || ("group "+(groups.length+1))).trim();
  groups.push({name, members:[...selecting], color: PAL[groups.length % PAL.length]});
  selecting.clear(); document.getElementById("gname").value="";
  renderGroups(); drawSub(); paintSelecting();
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
    row.querySelector("input").addEventListener("change", e => { gr.name=e.target.value; drawSub(); });
    row.querySelector("button").addEventListener("click", () => { groups.splice(k,1); renderGroups(); drawSub(); paintSelecting(); });
    box.appendChild(row);
  });
}
function exportGroups() {
  const out = groups.map(gr => ({name: gr.name,
    nodes: gr.members.map(i => ({label:N[i].short, layer:N[i].layer, position:N[i].pos, feature_idx:N[i].f}))}));
  const blob = new Blob([JSON.stringify(out,null,2)], {type:"application/json"});
  const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = "groups.json"; a.click();
}
document.getElementById("mk").addEventListener("click", makeGroup);
document.getElementById("clr").addEventListener("click", () => { selecting.clear(); paintSelecting(); });
document.getElementById("exp").addEventListener("click", exportGroups);

// ---------- pan / zoom ----------
function panzoom(svg) {
  let vb={x:0,y:0,w:D.w,h:D.h}, pan=false, sx=0, sy=0;
  function set(){ svg.setAttribute("viewBox",`${vb.x} ${vb.y} ${vb.w} ${vb.h}`); }
  svg.addEventListener("wheel", e => { e.preventDefault(); const k=e.deltaY>0?1.1:0.9;
    const r=svg.getBoundingClientRect(); const mx=(e.clientX-r.left)/r.width, my=(e.clientY-r.top)/r.height;
    const nw=vb.w*k, nh=vb.h*k; vb.x+=(vb.w-nw)*mx; vb.y+=(vb.h-nh)*my; vb.w=nw; vb.h=nh; set(); }, {passive:false});
  svg.addEventListener("mousedown", e => { if(e.target===svg || e.target.tagName==="line"){ pan=true; sx=e.clientX; sy=e.clientY; } });
  window.addEventListener("mousemove", e => { if(!pan) return; const r=svg.getBoundingClientRect();
    vb.x-=(e.clientX-sx)*(vb.w/r.width); vb.y-=(e.clientY-sy)*(vb.h/r.height); sx=e.clientX; sy=e.clientY; set(); });
  window.addEventListener("mouseup", () => pan=false);
}
panzoom(g); panzoom(sg);
g.addEventListener("click", () => { selected=null; clearEdges(); nodeEls.forEach(el=>el.classList.remove("dim")); paintSelecting(); });

renderGroups(); drawSub();
</script>
</body>
</html>"""
