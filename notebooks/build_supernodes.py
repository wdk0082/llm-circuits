"""Stage 2 (CPU) of the supernode pipeline: SEMANTIC supernode selection + review files.

Scans ALL feature nodes of the pruned graphs persisted by ``build_supernode_inputs.py``
and proposes paper-aligned supernodes by **label/example semantics** (+ operand-grid
class for addition) — influence is recorded as evidence only, never used to select.
Paper reference (notes/biology_digest.md): supernodes are small hand-curated clusters
(antonym=6, synonym=6, small=3, hot=3, FR-detect=3, ZH-detect=5), disjoint within an
experiment, named after concepts (Fig A2 / B1-B5).

Outputs:
* ``notebooks/supernodes/{multilingual,addition}_<size>.json`` — the reviewable
  proposal files (committed). Every member carries its evidence inline (top logits,
  an activation-example snippet, grid class, activation, influence) plus a ``review``
  field (``"proposed"`` until a human flips it). Top-level ``"approved": false`` —
  the notebook loader refuses unapproved files.
* ``artifacts/supernode_inputs/<size>/review_<graph>.html`` — reviewer explorer pages
  with the proposed supernodes pre-loaded as groups (click a member to see its label,
  logits, and activation examples).

Hard rules: ≤ MAX_MEMBERS per supernode (paper-small); disjoint within each task's
supernode set (same graph+position+feature can belong to one supernode only) — the
script fails loudly on violations. Overflow candidates (passed the semantic rule but
lost the size cut) are kept in ``overflow`` for the reviewer to swap in.

Run:  uv run python notebooks/build_supernodes.py --size 4b
Check committed files only:  uv run python notebooks/build_supernodes.py --check
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import addition_helper as A
import numpy as np

from llm_circuits.circuits.graph_explorer import render_graph_explorer_html
from llm_circuits.settings import artifacts_dir

SUPERNODE_DIR = Path(__file__).resolve().parent / "supernodes"
MAX_MEMBERS = 6

# ---------------------------------------------------------------------------
# Concept lexicons (multilingual). A node matches a concept when any of its label's
# top-logit tokens (lowercased, stripped) contains one of these strings.
# ---------------------------------------------------------------------------
CONCEPTS = {
    "opposite": ["opposite", "contrary", "antonym", "contraire", "inverse", "反义", "相反", "反"],
    "synonym": ["synonym", "synonyme", "同义", "近义", "equivalent"],
    # Qwen3's synonym graph carries no operation-labeled features (checked 2026-07-11);
    # its final-position features are ANSWER-side (say small/tiny). The donor supernode
    # is therefore selected by the synonym-mode answer lexicon, honestly named.
    "synonym_answer": [
        "small",
        "tiny",
        "little",
        "petit",
        "小",
        "微",
        "micro",
        "similar",
        "comparatively",
        "relatively",
        "相似",
        "mini",
    ],
    "small": ["small", "tiny", "little", "petit", "小", "微", "mini"],
    "hot": ["hot", "heat", "warm", "chaud", "热", "烫"],
    "large": ["large", "big", "grand", "huge", "大", "巨大", "bigger", "larger", "giant"],
    "cold": ["cold", "froid", "冷", "cool", "chill"],
}

_CJK = re.compile(r"[一-鿿]")
_FR_MARK = re.compile(r"[àâçéèêëîïôùûüœÀÂÇÉÈÊËÎÏÔÙÛÜŒ]")
_FR_WORDS = re.compile(r"\b(le|la|les|des|une|est|dans|pour|avec|qui|pas|sur)\b", re.I)


def text_lang(s: str) -> str:
    """Crude language of a snippet: zh (any CJK), fr (diacritics or ≥2 stopwords), en."""
    if _CJK.search(s):
        return "zh"
    if _FR_MARK.search(s) or len(_FR_WORDS.findall(s)) >= 2:
        return "fr"
    return "en"


def top_logit_strings(label: dict | None) -> list[str]:
    if not label:
        return []
    return [str(t).strip().lower() for t in (label.get("top_logits") or [])[:10]]


def matches_concept(label: dict | None, concept: str) -> str | None:
    """OUTPUT-side match: the concept appears in the feature's top logits (what the
    feature PROMOTES). Returns the matching top-logit string, or None."""
    for tok in top_logit_strings(label):
        for kw in CONCEPTS[concept]:
            if kw.lower() in tok:
                return tok
    return None


def peak_token_match(label: dict | None, concept: str, n: int = 6) -> tuple[int, int, str | None]:
    """INPUT-side match: does the PEAK-ACTIVATION token of the max-activating examples
    contain a concept word (i.e. the feature fires ON the concept)? Scans up to ``n``
    examples across quantile groups; returns (hits, scanned, first matching token)."""
    hits, total, first = 0, 0, None
    for q in (label or {}).get("examples") or []:
        for item in q.get("items", []):
            toks = item.get("tokens", [])
            acts = [float(x) for x in item.get("acts", [])]
            if not toks or not acts:
                continue
            k = max(range(len(acts)), key=acts.__getitem__)
            tok = str(toks[k]).strip().lower()
            total += 1
            if any(kw.lower() in tok for kw in CONCEPTS[concept]):
                hits += 1
                first = first or tok
            if total >= n:
                break
        if total >= n:
            break
    return hits, total, first


def concept_match(label: dict | None, concept: str, side: str = "both") -> str | None:
    """Match a concept on the requested side(s); returns a provenance string or None.

    ``side``: "output" (top logits — say-features), "input" (example peak tokens —
    features that fire ON the concept word, the paper's input supernodes), or "both".
    Input-side requires a majority of scanned examples (>=2 hits) to peak on the concept.
    """
    if side in ("output", "both"):
        tok = matches_concept(label, concept)
        if tok:
            return f"top_logits:{tok!r} (output-side)"
    if side in ("input", "both"):
        hits, total, tok = peak_token_match(label, concept)
        if total and hits >= max(2, total // 2):
            return f"peak_token:{tok!r} (input-side, {hits}/{total} examples)"
    return None


def example_snippet(label: dict | None, max_len: int = 90) -> str:
    """Peak-token-highlighted snippet from the top activation example."""
    for q in (label or {}).get("examples") or []:
        for item in q.get("items", []):
            toks, acts = item.get("tokens", []), [float(x) for x in item.get("acts", [])]
            if not toks:
                continue
            k = max(range(len(acts)), key=lambda j: acts[j]) if acts else 0
            parts = ["⟦" + t + "⟧" if j == k else t for j, t in enumerate(toks)]
            return "".join(parts).replace("\n", "\\n")[:max_len]
    return ""


def example_langs(label: dict | None, n: int = 6) -> list[str]:
    """Languages of the first n example snippets (across quantile groups)."""
    langs = []
    for q in (label or {}).get("examples") or []:
        for item in q.get("items", []):
            s = "".join(item.get("tokens", []))
            if s.strip():
                langs.append(text_lang(s))
            if len(langs) >= n:
                return langs
    return langs


# ---------------------------------------------------------------------------
# Graph access
# ---------------------------------------------------------------------------


def op_word_positions(gd: dict, word: str) -> set[int]:
    """Token positions covering the first occurrence of ``word`` in the prompt
    (multi-token words like 'contra|ire' and '反|义|词' span several positions)."""
    toks = gd["tokens"]
    text = "".join(toks)
    i = text.find(word)
    if i < 0:
        return set()
    spans, j = set(), 0
    for p, t in enumerate(toks):
        if j + len(t) > i and j < i + len(word):
            spans.add(p)
        j += len(t)
    return spans


def load_inputs(size: str):
    root = artifacts_dir() / "supernode_inputs" / size
    if not root.exists():
        raise SystemExit(f"{root} missing — run build_supernode_inputs.py --size {size} first")
    manifest = json.loads((root / "manifest.json").read_text())
    graphs = {
        p.stem.removeprefix("graph_"): json.loads(p.read_text())
        for p in sorted(root.glob("graph_*.json"))
    }
    return root, manifest, graphs


def feature_nodes(gd: dict, position: int | None = None):
    for n in gd["nodes"]:
        if n["node_type"] != "feature":
            continue
        if position is not None and n["position"] != position:
            continue
        yield n


def member_entry(node: dict, *, matched: str, grid_class: str | None = None, note: str = ""):
    e = {
        "layer": node["layer"],
        "feature": node["feature_idx"],
        "position": node["position"],
        "act": round(float(node.get("activation", 0.0)), 4),
        "influence": round(float(node.get("influence", 0.0)), 6),
        "evidence": {
            "matched": matched,
            "top_logits": (node.get("label") or {}).get("top_logits", [])[:8],
            "example": example_snippet(node.get("label")),
            "grid_class": grid_class,
        },
        "review": "proposed",
    }
    if note:
        e["review_note"] = note
    return e


def cap_members(cands: list[dict], key=lambda m: m["act"]) -> tuple[list[dict], list[dict]]:
    """Rank candidates (default: by activation) and split at MAX_MEMBERS."""
    ranked = sorted(cands, key=key, reverse=True)
    return ranked[:MAX_MEMBERS], ranked[MAX_MEMBERS:]


def supernode(name, paper_name, role, graph, position, members, overflow, note=""):
    sn = {
        "name": name,
        "paper_name": paper_name,
        "role": role,
        "graph": graph,
        "position": position,
        "members": members,
    }
    if overflow:
        sn["overflow"] = overflow
    if note:
        sn["note"] = note
    return sn


# ---------------------------------------------------------------------------
# Multilingual selection
# ---------------------------------------------------------------------------

LANGS = ("en", "fr", "zh")


def build_multilingual(size: str, root: Path, manifest: dict, graphs: dict) -> dict:
    sns: list[dict] = []
    mg = manifest["graphs"]

    def final(name):
        return mg[name]["final_position"]

    def shared_concept(concept, names, position_of, min_graphs=2, side="both"):
        """(L,f) matching `concept` (on ``side``) in >= min_graphs of `names`.

        ``position_of(name)`` scopes the scan (None = every position — the paper's
        supernodes are node SETS wherever they live). Per-graph activations and node
        positions are recorded so interventions can steer each member at its own
        node position per recipient graph.
        """
        hits: dict[tuple[int, int], dict] = {}
        seen_in: dict[tuple[int, int], list[str]] = defaultdict(list)
        for name in names:
            for n in feature_nodes(graphs[name], position_of(name)):
                m = concept_match(n.get("label"), concept, side)
                if not m:
                    continue
                key = (n["layer"], n["feature_idx"])
                if name not in seen_in[key]:
                    seen_in[key].append(name)
                if key not in hits:
                    hits[key] = member_entry(n, matched=m)
                    hits[key]["acts"] = {}
                    hits[key]["positions"] = {}
                prev = hits[key]["acts"].get(name, float("-inf"))
                if float(n["activation"]) > prev:
                    hits[key]["acts"][name] = round(float(n["activation"]), 4)
                    hits[key]["positions"][name] = n["position"]
        return [dict(hits[k], graphs=seen_in[k]) for k in hits if len(seen_in[k]) >= min_graphs]

    ant_names = [f"antonym_{lg}" for lg in LANGS]
    raw_ant_names = [f"raw_antonym_{lg}" for lg in LANGS]

    # --- operation swap: antonym (multilingual) source + synonym donor ----------------
    # Chat and raw scans seed the SAME paper name: identically-named groups merge at
    # ingestion into one supernode — the joint over ALL prompts (review convention).
    ant_keys: set[tuple[int, int]] = set()
    for names, syn_graph in (
        (ant_names, "synonym_en"),
        (raw_ant_names, "raw_synonym_en"),
    ):
        # the antonym-operation supernode lives mid-graph (paper Fig B1: opposite+small
        # -> antonym), not only on the final token: scan every position, BOTH sides
        # (input-side = fires on "opposite"/"contraire"/反义; output-side = promotes them).
        cands = shared_concept("opposite", names, lambda _n: None, side="both")
        members, overflow = cap_members(cands)
        ant_keys |= {(m["layer"], m["feature"]) for m in members + overflow}
        note = (
            "operation-swap source; steered -5x (m=-6) at each member's own node"
            " position per recipient graph"
        )
        sns.append(
            supernode(
                "antonym (multilingual)",
                "antonym",
                "source",
                ";".join(names),
                "member-positions",
                members,
                overflow,
                note=note,
            )
        )
        # paper-faithful donor: synonym OPERATION features — they fire ON the word
        # "synonym"/"synonymous" (input-side peaks) at the operation-word token of the
        # donor prompt; found only once the input side was searched (2026-07-11 review).
        op_best: dict[tuple[int, int], dict] = {}
        for n in feature_nodes(graphs[syn_graph]):
            m = concept_match(n.get("label"), "synonym", "both")
            if not m:
                continue
            key = (n["layer"], n["feature_idx"])
            entry = member_entry(n, matched=m)
            if key not in op_best or entry["act"] > op_best[key]["act"]:
                op_best[key] = entry
        members, overflow = cap_members(list(op_best.values()))
        note = (
            "operation-swap donor (paper-faithful): synonym-OPERATION features from the"
            " EN synonym prompt, injected at +6x the stored act. They live at the"
            " 'synonym' word token of the donor prompt; on the recipient, inject at the"
            " operation-word position ('opposite'/'contraire'/反义词)."
        )
        if not members:
            note += (
                " EMPTY: the pruned raw synonym graph keeps no synonym-operation"
                " features (raw-graph finding) — say small (multilingual) below is the"
                " only raw answer-side alternative."
            )
        sns.append(
            supernode(
                "synonym (multilingual)",
                "synonym",
                "donor",
                syn_graph,
                "member-positions",
                members,
                overflow,
                note=note,
            )
        )
        syn_cands = [
            member_entry(
                n,
                matched=f"top_logits:{matches_concept(n.get('label'), 'synonym_answer')!r}",
            )
            for n in feature_nodes(graphs[syn_graph], final(syn_graph))
            if matches_concept(n.get("label"), "synonym_answer")
            and not matches_concept(n.get("label"), "large")
        ]
        members, overflow = cap_members(syn_cands)
        sns.append(
            supernode(
                "say small (multilingual)",
                "say small",
                "readout",
                syn_graph,
                "final",
                members,
                overflow,
                note="operation-swap readout: say small/tiny features at the synonym"
                " prompt's final position (they can also serve as the v1-style"
                " answer-side donor; the paper-faithful donor is synonym (multilingual)"
                " above).",
            )
        )

    # --- operand swap: small (multilingual) source + hot donor + say-cold readout -----
    cands = shared_concept(
        "small",
        ant_names + raw_ant_names,
        lambda n: mg[n]["operand_position"],
        side="both",
    )
    members, overflow = cap_members(cands)
    sns.append(
        supernode(
            "small (multilingual)",
            "small",
            "source",
            ";".join(ant_names + raw_ant_names),
            "operand",
            members,
            overflow,
            note="operand-swap source; steered -0.5x (m=-1.5). Input-side matches (peak"
            " example token = small/petit/...) are the paper's operand-feature analogue;"
            " output-side matches at the operand token are say-small-ish — extra scrutiny"
            " at review",
        )
    )
    hot_names = [g for g in ("hot_en", "raw_hot_en") if g in graphs]
    cands = shared_concept(
        "hot", hot_names, lambda n: mg[n]["operand_position"], min_graphs=1, side="both"
    )
    members, overflow = cap_members(cands)
    sns.append(
        supernode(
            "hot (multilingual)",
            "hot",
            "donor",
            ";".join(hot_names),
            "operand",
            members,
            overflow,
            note="operand-swap donor; injected value = +1.5x the stored act (per donor"
            " graph — the raw page is the paper's exact prompt format)",
        )
    )
    cands = shared_concept("cold", hot_names, final, min_graphs=1, side="output")
    members, overflow = cap_members(cands)
    sns.append(
        supernode(
            "say cold (multilingual)",
            "say cold",
            "readout",
            ";".join(hot_names),
            "final",
            members,
            overflow,
        )
    )

    # --- say large: multilingual (>=2 graphs) vs language-specific (exactly 1) --------
    for names in (ant_names, raw_ant_names):
        shared = shared_concept("large", names, final, min_graphs=2)
        members, overflow = cap_members(shared)
        sns.append(
            supernode(
                "say large (multilingual)",
                "say large",
                "readout",
                ";".join(names),
                "final",
                members,
                overflow,
            )
        )
        shared_keys = {(m["layer"], m["feature"]) for m in members + overflow}
        for lg in LANGS:
            name = names[LANGS.index(lg)]
            only = []
            for n in feature_nodes(graphs[name], final(name)):
                m = matches_concept(n.get("label"), "large")
                if not m or (n["layer"], n["feature_idx"]) in shared_keys:
                    continue
                entry = member_entry(n, matched=f"top_logits:{m!r}")
                # script sanity: a {lang}-specific say-large whose only match is a
                # different script (e.g. a CJK token on the FR graph) is suspect
                tok_lang = text_lang(m)
                if (lg == "zh") != (tok_lang == "zh"):
                    entry["review_note"] = (
                        f"matched token {m!r} is {tok_lang}-script on the {lg}-specific"
                        " supernode — likely junk/mixed feature, verify the example"
                    )
                only.append(entry)
            members_l, overflow_l = cap_members(only)
            sns.append(
                supernode(
                    f"say large ({lg})",
                    f"say large ({lg})",
                    "readout",
                    name,
                    "final",
                    members_l,
                    overflow_l,
                )
            )

    # --- language detectors (graph-first; Fig B5 source/donor) ------------------------
    # The paper's detectors sit at the final open-quote token, early layers. Qwen3's
    # 0.95 graphs have NO layer<12 nodes at that position, so the seed keeps only the
    # semantic filter (language-pure examples) and prefers the earliest layers instead
    # of hard-cutting — the human review decides what actually counts as a detector.
    detect_cands: dict[str, list[dict]] = {}
    for lg in LANGS:
        name = f"raw_antonym_{lg}"
        cands = []
        for n in feature_nodes(graphs[name], final(name)):
            langs = example_langs(n.get("label"))
            if not langs:
                continue
            frac = langs.count(lg) / len(langs)
            if frac >= 0.7:
                cands.append(
                    member_entry(
                        n,
                        matched=f"examples:{langs.count(lg)}/{len(langs)} {lg}",
                        note="graph-first detector (language-pure examples;"
                        " earliest layers preferred)",
                    )
                )
        detect_cands[lg] = cands
    # language-unique: drop (L,f) appearing as a candidate for >=2 languages
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for lg in LANGS:
        for m in detect_cands[lg]:
            counts[(m["layer"], m["feature"])] += 1
    for lg in LANGS:
        uniq = [m for m in detect_cands[lg] if counts[(m["layer"], m["feature"])] == 1]
        members, overflow = cap_members(uniq, key=lambda m: (-m["layer"], m["act"]))
        sns.append(
            supernode(
                f"quote ({lg})",
                f"open-quote-in-{lg}",
                "source+donor",
                f"raw_antonym_{lg}",
                "final",
                members,
                overflow,
                note="language-swap source/donor seed (paper: 'quote (lang-specific)';"
                " graph-first on the raw open-quote graphs; language-pure examples,"
                " earliest layers first — Qwen3's quote-position nodes all sit above"
                " L12, unlike the paper's early detectors). The review decides.",
            )
        )

    # --- opposite (lang-specific): the operation WORD's own features (paper Fig B1:
    # 'opposite (lang-specific)' feeds 'antonym (multilingual)') --------------------
    op_word = {"en": "opposite", "fr": "contraire", "zh": "反义词"}
    for lg in LANGS:
        best: dict[tuple[int, int], dict] = {}
        for name in (f"antonym_{lg}", f"raw_antonym_{lg}"):
            positions = op_word_positions(graphs[name], op_word[lg])
            for n in feature_nodes(graphs[name]):
                key = (n["layer"], n["feature_idx"])
                if n["position"] not in positions or key in ant_keys:
                    continue
                e = member_entry(n, matched=f"at operation-word token {op_word[lg]!r}")
                if key not in best or e["act"] > best[key]["act"]:
                    best[key] = e
        members, overflow = cap_members(list(best.values()))
        sns.append(
            supernode(
                f"opposite ({lg})",
                f"opposite ({lg})",
                "input",
                f"antonym_{lg};raw_antonym_{lg}",
                "member-positions",
                members,
                overflow,
                note="language-specific operation-word features (upstream evidence;"
                " not steered in the paper's three swaps). Features already seeded"
                " into antonym (multilingual) are excluded — supernodes are disjoint.",
            )
        )

    return {
        "task": "multilingual",
        "size": size,
        "built_from": f"{root}@{manifest.get('git', '?')}",
        "max_members": MAX_MEMBERS,
        "supernodes": sns,
        "approved": False,
    }


# ---------------------------------------------------------------------------
# Addition selection (grid classes are the evidence)
# ---------------------------------------------------------------------------


def build_addition(size: str, root: Path, manifest: dict, graphs: dict) -> dict:
    mg = manifest["graphs"]
    z = np.load(root / "grids.npz")
    a_vals, b_vals = z["a_vals"].tolist(), z["b_vals"].tolist()

    reports: dict[str, dict[tuple[int, int], dict]] = {}
    for probe in ("first", "ones", "last"):
        feats = [tuple(x) for x in z[f"{probe}_features"]]
        arr = z[f"{probe}_grids"]
        reports[probe] = {
            f: A.periodicity_report(arr[i], a_vals, b_vals) for i, f in enumerate(feats)
        }

    def rep(probe, node):
        return reports[probe].get((node["layer"], node["feature_idx"]))

    def classed(graph_name, position, probe, predicate, matched_fmt):
        # dedup by (layer, feature): with position=None the same feature can appear as a
        # node at several positions — keep the highest-activation instance.
        best: dict[tuple[int, int], dict] = {}
        for n in feature_nodes(graphs[graph_name], position):
            r = rep(probe, n)
            if r is None or not predicate(r):
                continue
            key = (n["layer"], n["feature_idx"])
            entry = member_entry(n, matched=matched_fmt(r), grid_class=str(r["label"]))
            if key not in best or entry["act"] > best[key]["act"]:
                best[key] = entry
        return list(best.values())

    sns: list[dict] = []
    ones, first = mg["ones"], mg["first"]
    dp = ones["digit_positions"]
    fin_ones, fin_first = ones["final_position"], first["final_position"]
    fin_donor = mg["donor99"]["final_position"]

    def lab(r):
        return str(r["label"])

    # input supernodes: grid class + residue at the operand-digit NODE positions
    # (the v1 notebook's reviewed predicates: label startswith (mod10-a|lookup) with
    # a_top == a%10, from the a-digit position pool — here the pool is every graph node
    # at those positions, layer < n//4 like the v1 pools).
    def at_positions(graph_name, positions, max_layer=9):
        seen = set()
        for pos in positions:
            for n in feature_nodes(graphs[graph_name], pos):
                if n["layer"] >= max_layer:
                    continue
                key = (n["layer"], n["feature_idx"], n["position"])
                if key not in seen:
                    seen.add(key)
                    yield n

    def input_class(positions, pred, steer_position):
        best: dict[tuple[int, int], dict] = {}
        for n in at_positions("ones", positions):
            r = rep("first", n)
            if r is None or not pred(r):
                continue
            key = (n["layer"], n["feature_idx"])
            entry = member_entry(n, matched=f"grid:{lab(r)}", grid_class=lab(r))
            if key not in best or entry["act"] > best[key]["act"]:
                best[key] = entry
        return list(best.values())

    a_res, b_res = 46 % 10, 49 % 10
    for name, paper, positions, steer_pos, pred in (
        (
            "input _6",
            "_6",
            dp["a_digits"],
            dp["a_digits"][-1],
            lambda r: lab(r).startswith(("mod10-a", "lookup")) and r.get("a_top") == a_res,
        ),
        (
            "input _9",
            "_9",
            dp["b_digits"],
            dp["b_digits"][-1],
            lambda r: lab(r).startswith(("mod10-b", "lookup")) and r.get("b_top") == b_res,
        ),
        (
            "magnitude ~46",
            "~36-analogue",
            dp["a_digits"] + dp["b_digits"],
            None,
            lambda r: lab(r).startswith("band-a"),
        ),
        (
            "magnitude ~49",
            "~59-analogue",
            dp["a_digits"] + dp["b_digits"],
            None,
            lambda r: lab(r).startswith("band-b"),
        ),
    ):
        cands = input_class(positions, pred, steer_pos)
        members, overflow = cap_members(cands)
        sns.append(supernode(name, paper, "source", "ones", steer_pos, members, overflow))

    # answer-side classes (probe="ones": the predict-ones position)
    lookup_cands = classed(
        "ones",
        fin_ones,
        "ones",
        lambda r: lab(r).startswith("lookup"),
        lambda r: f"grid:{lab(r)}",
    )

    def add_on_pair(cands, probe, a, b):
        """Evidence: activation on the studied pair as a fraction of the grid max —
        multi-pair lattices count as on-concept only if the pair is in their receptive
        field; auto-flag members below 0.3."""
        feats = [tuple(int(v) for v in x) for x in z[f"{probe}_features"]]
        idx = {f: i for i, f in enumerate(feats)}
        arr = z[f"{probe}_grids"]
        for m in cands:
            i = idx.get((m["layer"], m["feature"]))
            if i is None:
                continue
            g = arr[i]
            mx = float(np.nanmax(g)) or 1.0
            frac = float(g[a_vals.index(a), b_vals.index(b)]) / mx
            m["evidence"]["on_pair_frac_of_max"] = round(frac, 3)
            if frac < 0.3:
                m["review_note"] = (
                    m.get("review_note", "")
                    + f" on-pair activation only {frac:.0%} of grid max — verify the"
                    " receptive field really includes the studied pair"
                ).strip()

    add_on_pair(lookup_cands, "ones", 46, 49)
    members, overflow = cap_members(lookup_cands)
    sns.append(
        supernode(
            "lookup (6,9-class)",
            "_6+_9",
            "source",
            "ones",
            "final",
            members,
            overflow,
            note="active multi-pair lookup lattices whose receptive fields include (6,9)",
        )
    )
    sum_cands = classed(
        "ones",
        fin_ones,
        "ones",
        lambda r: lab(r).startswith("mod10-sum(r5)"),
        lambda r: f"grid:{lab(r)}",
    )
    members, overflow = cap_members(sum_cands)
    sns.append(
        supernode(
            "sum = _5",
            "sum = _5",
            "readout",
            "ones",
            "final",
            members,
            overflow,
            note="residue-5 sum features (95 % 10); other-residue mod10-sum features that"
            " are also active on this prompt are deliberately excluded",
        )
    )

    lowprec_cands = classed(
        "first",
        fin_first,
        "last",
        lambda r: lab(r).startswith("magnitude-diag"),
        lambda r: f"grid:{lab(r)}",
    )
    members, overflow = cap_members(lowprec_cands)
    sns.append(
        supernode(
            "sum ~95 (low precision)",
            "sum ~92-analogue",
            "readout",
            "first",
            "final",
            members,
            overflow,
        )
    )

    donor_cands = classed(
        "donor99",
        fin_donor,
        "ones",
        lambda r: lab(r).startswith("lookup") and r.get("a_top") == 9 and r.get("b_top") == 9,
        lambda r: f"grid:{lab(r)} a_top=9 b_top=9",
    )
    add_on_pair(donor_cands, "ones", 49, 49)
    members, overflow = cap_members(donor_cands)
    sns.append(
        supernode(
            "lookup (9,9) donors",
            "_9+_9",
            "donor",
            "donor99",
            "final",
            members,
            overflow,
            note="swap donor; injected value = +1x the stored act (paper +1x)",
        )
    )

    # the paper's add-function class (`add _9`, `add ~57`): one-operand stripe/band
    # signatures read at the answer position — a MEASURED NEGATIVE on Qwen3-4B (v1,
    # third session). The supernode stays in the file so the negative is a live,
    # reviewable scan result instead of a retired figure.
    def addf_pred(r):
        # PAIR-CONSISTENT one-operand signatures only: for 46+49 the paper's class
        # would be `add _6`/`add _9` (the operands' residues) or `add ~46`/`add ~49`
        # (their magnitudes). Unconstrained stripe/band classes at the answer position
        # are dominated by junk-labeled textures (first scan: 13 candidates, all with
        # off-pair classes like band-a(~6) or mod10-a(r9) — grid-class false positives).
        label = lab(r)
        return (
            label.startswith(("mod10-a(r6)", "mod10-b(r9)"))
            or (label.startswith("band-a") and 40 <= r.get("a_mean", -99) <= 52)
            or (label.startswith("band-b") and 43 <= r.get("b_mean", -99) <= 55)
        )

    addf_cands = classed("ones", fin_ones, "ones", addf_pred, lambda r: f"grid:{lab(r)}")
    for m in addf_cands:
        m["review_note"] = (
            "add-function candidate — verify labels/examples read as add-semantics and"
            " the grid visually shows a clean one-operand stripe/band (v1's single"
            " nominal flag was rejected as a diffuse texture)"
        )
    members, overflow = cap_members(addf_cands)
    sns.append(
        supernode(
            "add function (hunt)",
            "add _9 / add ~57",
            "annotation",
            "ones",
            "final",
            members,
            overflow,
            note="EXPECTED (EFFECTIVELY) EMPTY — the paper's add-function class at the"
            " answer position is a measured negative on Qwen3-4B: the pair-consistent"
            " scan yields only junk-labeled candidates. The operator-token"
            " mostly-active signature remains the only add-function-adjacent finding.",
        )
    )

    # exact-value inputs (the paper's `36`/`59` nodes): cross class at digit positions
    for name, paper, val in (
        ("input 46 (exact)", "36-analogue", 46),
        ("input 49 (exact)", "59-analogue", 49),
    ):
        cands = input_class(
            dp["a_digits"] + dp["b_digits"],
            lambda r, v=val: lab(r) == f"exact-cross({v})",
            None,
        )
        members, overflow = cap_members(cands)
        sns.append(
            supernode(
                name,
                paper,
                "annotation",
                "ones",
                None,
                members,
                overflow,
                note="exact-value input class (fires whenever either operand IS the"
                " value); descriptive — no experiment steers it",
            )
        )

    # magnitude-lookup regions (the paper's wide/narrow `~36+~60` class): the
    # first-digit circuit's local blobs feeding the low-precision sum
    region_cands = classed(
        "first",
        fin_first,
        "last",
        lambda r: lab(r).startswith("region("),
        lambda r: f"grid:{lab(r)}",
    )
    members, overflow = cap_members(region_cands)
    sns.append(
        supernode(
            "magnitude lookup (~46+~49)",
            "~36+~60",
            "annotation",
            "first",
            "final",
            members,
            overflow,
            note="2-D localized non-repeating blobs — the paper's wide/narrow"
            " magnitude-lookup class; descriptive — no experiment steers it",
        )
    )

    # polymer reuse: lookup members active at the polymer ones moment
    pol = json.loads((root / "polymer_ones_acts.json").read_text())
    active = {(int(L), int(i)): float(a) for L, i, a in pol["active_final"]}
    lookup_sn = next(s for s in sns if s["name"] == "lookup (6,9-class)")
    reuse_members = []
    for m in lookup_sn["members"]:
        k = (m["layer"], m["feature"])
        if k in active:
            mm = dict(m)
            mm["act"] = round(active[k], 4)
            mm["evidence"] = dict(
                m["evidence"], matched=m["evidence"]["matched"] + "; active at polymer ones moment"
            )
            reuse_members.append(mm)
    sns.append(
        supernode(
            "polymer-active lookups",
            "_6+_9 (reuse)",
            "source",
            "polymer-ones-moment",
            "final",
            reuse_members,
            [],
            note="members = lookup (6,9-class) ∩ active at 'K. Whang…, 199' predicting 5; "
            "acts are the polymer-moment activations",
        )
    )

    return {
        "task": "addition",
        "size": size,
        "built_from": f"{root}@{manifest.get('git', '?')}",
        "max_members": MAX_MEMBERS,
        "supernodes": sns,
        "approved": False,
    }


# ---------------------------------------------------------------------------
# Disjointness + validation + reviewer HTMLs
# ---------------------------------------------------------------------------


# First-pass reviewer flags (2026-07-11 evidence read): members I would question but
# do not decide on — the review gate owns the verdicts. Keyed (supernode, layer, feature).
MANUAL_FLAGS: dict = {}  # (supernode, layer, feature) -> reviewer note, if ever needed


# Delegated review record (2026-07-11; the user waived the manual gate for the
# grid-principled addition selection). Re-applied on every emit so selector re-runs
# reproduce the reviewed state instead of clobbering it. Multilingual stays unapproved.
REJECTED_MEMBERS = {
    ("addition", "lookup (6,9-class)", 33, 109892): (
        "on-pair activation 20% of grid max fails the >=30% receptive-field sanity;"
        " replaced by the top act-ranked overflow qualifier"
    ),
}
APPROVED_TASKS = {
    "addition": "grid-principled selection reviewed by delegation (user waived the"
    " manual gate); rejections in REJECTED_MEMBERS; all other members approved",
}


def apply_review_decisions(task: str, doc: dict) -> None:
    for sn in doc["supernodes"]:
        kept, rejected = [], []
        for m in sn["members"]:
            reason = REJECTED_MEMBERS.get((task, sn["name"], m["layer"], m["feature"]))
            if reason:
                m["review"] = "rejected"
                m["review_note"] = (m.get("review_note", "") + " REVIEW: " + reason).strip()
                rejected.append(m)
            else:
                kept.append(m)
        if rejected:
            promoted = sn.get("overflow", [])[: len(rejected)]
            sn["overflow"] = rejected + sn.get("overflow", [])[len(rejected) :]
            kept += promoted
        sn["members"] = kept
        if task in APPROVED_TASKS:
            for m in sn["members"]:
                m["review"] = "approved"
    if task in APPROVED_TASKS:
        doc["approved"] = True
        doc["review_log"] = f"2026-07-11: {APPROVED_TASKS[task]}"


def apply_manual_flags(doc: dict) -> None:
    for sn in doc["supernodes"]:
        for m in sn["members"]:
            note = MANUAL_FLAGS.get((sn["name"], m["layer"], m["feature"]))
            if note:
                m["review_note"] = (m.get("review_note", "") + " " + note).strip()


def check_disjoint(doc: dict) -> list[str]:
    """A feature@graph+position may belong to at most one supernode (members only)."""
    seen: dict[tuple, str] = {}
    clashes = []
    for sn in doc["supernodes"]:
        for m in sn["members"]:
            key = (sn["graph"], str(sn["position"]), m["layer"], m["feature"])
            if key in seen and seen[key] != sn["name"]:
                clashes.append(f"{key} in both {seen[key]!r} and {sn['name']!r}")
            seen[key] = sn["name"]
    return clashes


def validate(doc: dict, *, require_approved: bool) -> list[str]:
    errs = []
    for sn in doc["supernodes"]:
        if len(sn["members"]) > doc.get("max_members", MAX_MEMBERS):
            errs.append(f"{sn['name']}: {len(sn['members'])} members > cap")
        for m in sn["members"]:
            if m.get("review") == "rejected":
                errs.append(
                    f"{sn['name']}: rejected member L{m['layer']}f{m['feature']} still in members"
                )
    errs += check_disjoint(doc)
    if require_approved and not doc.get("approved"):
        errs.append("file not approved")
    return errs


def emit_review_htmls(root: Path, manifest: dict, graphs: dict, docs: list[dict]) -> None:
    per_graph: dict[str, list[dict]] = defaultdict(list)
    for doc in docs:
        for sn in doc["supernodes"]:
            for gname in str(sn["graph"]).split(";"):
                if gname not in graphs:
                    continue
                members = [(m["layer"], m["feature"]) for m in sn["members"]]
                if not members:
                    continue
                pos = sn["position"]
                # Only "final" and integer positions restrict resolution; markers like
                # "operand"/"member-positions" mean the (L,f) members pin the nodes.
                per_graph[gname].append(
                    {
                        "name": sn["name"],
                        "members": members,
                        "position": pos if pos == "final" or isinstance(pos, int) else None,
                    }
                )
    for gname, gspecs in per_graph.items():
        # Filename convention: review_chat_* / review_raw_* for the multilingual pairs
        # (the manifest's "raw" flag exists only for those); addition stays review_*.
        # The DUMP name (= export "example" label) is unprefixed either way.
        raw_flag = manifest["graphs"].get(gname, {}).get("raw")
        stem = f"chat_{gname}" if raw_flag is False else gname
        out_html = root / f"review_{stem}.html"
        render_graph_explorer_html(
            graphs[gname],
            out_html,
            labels=[gname],  # exports carry this as "example" -> unambiguous mapping
            title=f"review {stem}",
            groups=gspecs,
        )
        print(f"  {out_html.name}: {len(gspecs)} seeded supernodes")


# Metadata for group names the notebooks know how to drive — the paper's own
# vocabulary (Fig B1 skeleton: opposite/quote/say-large lang-specific; antonym/small/
# large/say-large multilingual; synonym + hot from the EN donor prompts). The group
# NAME is the merge key across pages: identically-named groups union into one
# supernode (chat + raw alike), language-specific groups carry their (en/fr/zh).
# Unknown names ingest fine (role "custom") but need wiring before they steer.
_LGS = ("en", "fr", "zh")
ROLE_BY_NAME = {
    "antonym (multilingual)": ("antonym", "source"),
    "synonym (multilingual)": ("synonym", "donor"),
    "small (multilingual)": ("small", "source"),
    "hot (multilingual)": ("hot", "donor"),
    "large (multilingual)": ("large", "concept"),
    "cold (multilingual)": ("cold", "concept"),
    **{
        f"say {w} (multilingual)": (f"say {w}", "readout")
        for w in ("large", "small", "hot", "cold")
    },
    **{
        f"say {w} ({lg})": (f"say {w} ({lg})", "readout")
        for w in ("large", "small", "hot", "cold")
        for lg in _LGS
    },
    **{f"opposite ({lg})": (f"opposite ({lg})", "input") for lg in _LGS},
    **{f"quote ({lg})": (f"open-quote-in-{lg}", "source+donor") for lg in _LGS},
}


def ingest_exports(size, files, root, manifest, graphs) -> None:
    """Write the multilingual supernode file FROM the explorer 'Export groups' JSONs.

    The exports are the review: every listed node was hand-picked in the UI, so members
    arrive ``review: "approved"`` and the file ships ``approved: true``. Same-named
    groups across several graph pages merge into one multi-graph supernode (per-graph
    ``acts``/``positions`` recorded — donors take value = mult x their own graph's act;
    sources steer at each member's node position per recipient). Evidence is
    auto-filled from the graph dumps. Disjointness within the final set is enforced.
    """
    merged: dict[str, dict] = {}
    for f in files:
        exp = json.loads(Path(f).read_text())
        gname = str(exp.get("example", "")).strip()
        if gname not in graphs:
            raise SystemExit(
                f"{f}: example label {gname!r} is not a dumped graph — re-export from"
                f" the CURRENT review pages (known: {sorted(graphs)})"
            )
        nodes_at = {
            (n["layer"], n["feature_idx"], n["position"]): n for n in feature_nodes(graphs[gname])
        }
        claimed: dict[tuple, str] = {}
        for grp in exp.get("groups", []):
            name = str(grp["name"]).strip()
            for nd in grp.get("nodes", []):
                k = (int(nd["layer"]), int(nd["feature_idx"]), nd.get("position"))
                if k in claimed and claimed[k] != name:
                    raise SystemExit(
                        f"{f}: node L{k[0]}f{k[1]}@p{k[2]} is in both {claimed[k]!r} and"
                        f" {name!r} — supernodes must be disjoint; fix in the UI and"
                        " re-export"
                    )
                claimed[k] = name
            sn = merged.setdefault(
                name,
                {
                    "name": name,
                    "paper_name": ROLE_BY_NAME.get(name, (name, "custom"))[0],
                    "role": ROLE_BY_NAME.get(name, (name, "custom"))[1],
                    "graphs": set(),
                    "members": {},
                },
            )
            sn["graphs"].add(gname)
            for nd in grp.get("nodes", []):
                key = (int(nd["layer"]), int(nd["feature_idx"]))
                node = nodes_at.get((key[0], key[1], nd.get("position")))
                if key not in sn["members"]:
                    if node is not None:
                        m = member_entry(node, matched="manual (explorer export)")
                    else:
                        m = {
                            "layer": key[0],
                            "feature": key[1],
                            "position": nd.get("position"),
                            "act": 0.0,
                            "influence": None,
                            "evidence": {
                                "matched": "manual (explorer export; not found in dump"
                                " at that position)",
                                "top_logits": [],
                                "example": "",
                                "grid_class": None,
                            },
                        }
                    m["source"] = "explorer-export"
                    m["review"] = "approved"
                    m["acts"], m["positions"] = {}, {}
                    sn["members"][key] = m
                m = sn["members"][key]
                if node is not None:
                    m["acts"][gname] = round(float(node["activation"]), 4)
                m["positions"][gname] = nd.get("position")

    # Paper supernodes are disjoint FEATURE sets: the same (layer, feature) may not
    # appear in two groups anywhere across the pages (per-page overlap is caught above;
    # this catches e.g. 'antonym' on the EN page vs 'opposite (fr)' on the FR page).
    owner: dict[tuple[int, int], str] = {}
    for sn in merged.values():
        for key in sn["members"]:
            if key in owner and owner[key] != sn["name"]:
                raise SystemExit(
                    f"feature L{key[0]}f{key[1]} is in both {owner[key]!r} and"
                    f" {sn['name']!r} (possibly on different pages) — supernodes are"
                    " disjoint feature sets; fix in the UI and re-export"
                )
            owner[key] = sn["name"]

    supernodes = []
    for sn in merged.values():
        members = sorted(sn["members"].values(), key=lambda m: -max(m["acts"].values() or [0]))
        note = "hand-selected in the explorer review pages (Export groups)"
        if sn["role"] == "custom":
            note += " — UNKNOWN group name: wire its role in the notebook before use"
        supernodes.append(
            supernode(
                sn["name"],
                sn["paper_name"],
                sn["role"],
                ";".join(sorted(sn["graphs"])),
                "member-positions",
                members,
                [],
                note=note,
            )
        )
    doc = {
        "task": "multilingual",
        "size": size,
        "built_from": f"{root}@{manifest.get('git', '?')}",
        "selection": "explorer-export",
        "source_files": [str(Path(f).name) for f in files],
        "supernodes": supernodes,
        "approved": True,
        "review_log": "selected and reviewed by hand in the explorer (Export groups);"
        " ingested by build_supernodes.py --from-exports",
    }
    errs = validate(doc, require_approved=True)
    if errs:
        raise SystemExit(
            "ingest: validation failed (fix the groups in the UI and re-export):\n  "
            + "\n  ".join(errs)
        )
    out = SUPERNODE_DIR / f"multilingual_{size}.json"
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    print(f"{out} written from {len(files)} exports:")
    for sn in supernodes:
        print(f"   {sn['name']:34s} {len(sn['members'])} members  [{sn['role']}]  {sn['graph']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="4b")
    ap.add_argument("--check", action="store_true", help="validate committed files only")
    ap.add_argument(
        "--from-exports",
        nargs="+",
        metavar="EXPORT_JSON",
        help="ingest explorer 'Export groups' JSONs (one per graph) and write the"
        " multilingual supernode file from them — the ONLY multilingual selection"
        " source; membership evidence is auto-filled from the graph dumps",
    )
    args = ap.parse_args()

    if args.check:
        errs_all = []
        for p in sorted(SUPERNODE_DIR.glob("*.json")):
            doc = json.loads(p.read_text())
            errs = validate(doc, require_approved=False)
            print(
                f"{p.name}: {'OK' if not errs else 'INVALID'}"
                + (f" approved={doc.get('approved')}" if not errs else "")
            )
            for e in errs:
                print("   -", e)
            errs_all += errs
        sys.exit(1 if errs_all else 0)

    root, manifest, graphs = load_inputs(args.size)
    SUPERNODE_DIR.mkdir(exist_ok=True)

    if args.from_exports:
        ingest_exports(args.size, args.from_exports, root, manifest, graphs)
        return

    docs = []
    # ADDITION: the grid-evidenced scan IS the selection (delegated review encoded).
    doc = build_addition(args.size, root, manifest, graphs)
    apply_manual_flags(doc)
    apply_review_decisions("addition", doc)
    errs = validate(doc, require_approved=False)
    if errs:
        raise SystemExit("addition: validation failed:\n  " + "\n  ".join(errs))
    out = SUPERNODE_DIR / f"addition_{args.size}.json"
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    docs.append(doc)
    print(f"{out} written:")
    for sn in doc["supernodes"]:
        print(f"   {sn['name']:34s} {len(sn['members'])} members")

    # MULTILINGUAL: the scan only SEEDS the review pages. The authoritative selection
    # is the human's explorer "Export groups" JSONs, ingested via --from-exports.
    seed = build_multilingual(args.size, root, manifest, graphs)
    docs.append(seed)
    print("\nmultilingual: scan used as review-page seeds only (no file written)")
    for sn in seed["supernodes"]:
        flag = "" if sn["members"] else "   <-- empty seed"
        print(f"   {sn['name']:34s} {len(sn['members'])} seeded{flag}")

    emit_review_htmls(root, manifest, graphs, docs)
    print(
        "\nNEXT (multilingual): adjust groups in the review_*.html pages, Export"
        "\ngroups per page, then: build_supernodes.py --from-exports <files...>"
    )


if __name__ == "__main__":
    main()
