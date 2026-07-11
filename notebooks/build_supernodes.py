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
from llm_circuits.transcoders.feature_labels import (
    load_feature_examples,
    load_feature_labels,
)
from llm_circuits.transcoders.registry import get_spec

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
    """Return the matching top-logit string, or None."""
    for tok in top_logit_strings(label):
        for kw in CONCEPTS[concept]:
            if kw.lower() in tok:
                return tok
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
    repo_id = get_spec(f"qwen3-{size}").transcoder_repo

    def final(name):
        return mg[name]["final_position"]

    def shared_concept(concept, names, position_of, min_graphs=2):
        """(L,f) matching `concept` in >= min_graphs of `names`.

        ``position_of(name)`` scopes the scan (None = every position — the paper's
        supernodes are node SETS wherever they live). Per-graph activations and node
        positions are recorded so interventions can steer each member at its own
        node position per recipient graph.
        """
        hits: dict[tuple[int, int], dict] = {}
        seen_in: dict[tuple[int, int], list[str]] = defaultdict(list)
        for name in names:
            for n in feature_nodes(graphs[name], position_of(name)):
                m = matches_concept(n.get("label"), concept)
                if not m:
                    continue
                key = (n["layer"], n["feature_idx"])
                if name not in seen_in[key]:
                    seen_in[key].append(name)
                if key not in hits:
                    hits[key] = member_entry(n, matched=f"top_logits:{m!r}")
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
    for tag, names, syn_graph in (
        ("", ant_names, "synonym_en"),
        ("raw ", raw_ant_names, "raw_synonym_en"),
    ):
        # the antonym-operation supernode lives mid-graph (paper Fig B1: opposite+small
        # -> antonym), not only on the final token: scan every position.
        cands = shared_concept("opposite", names, lambda _n: None)
        members, overflow = cap_members(cands)
        sns.append(
            supernode(
                f"{tag}antonym (multilingual)",
                "antonym",
                "source",
                ";".join(names),
                "member-positions",
                members,
                overflow,
                note="operation-swap source; steered -5x (m=-6) at each member's own node"
                " position per recipient graph",
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
                f"{tag}synonym (say-answer)",
                "synonym",
                "donor",
                syn_graph,
                "final",
                members,
                overflow,
                note="operation-swap donor; injected value = +6x the stored act. MODEL"
                " DIFFERENCE vs paper: Qwen3's synonym graph has no operation-labeled"
                " features — these are the synonym-mode ANSWER features (say small/tiny),"
                " selected by the synonym-answer lexicon; the paper's donor was an"
                " operation supernode.",
            )
        )

    # --- operand swap: small (multilingual) source + hot donor + say-cold readout -----
    cands = shared_concept("small", ant_names, lambda n: mg[n]["operand_position"])
    members, overflow = cap_members(cands)
    sns.append(
        supernode(
            "small (multilingual)",
            "small",
            "source",
            ";".join(ant_names),
            "operand",
            members,
            overflow,
            note="operand-swap source; steered -0.5x (m=-1.5)",
        )
    )
    hot_cands = [
        member_entry(n, matched=f"top_logits:{matches_concept(n.get('label'), 'hot')!r}")
        for n in feature_nodes(graphs["hot_en"], mg["hot_en"]["operand_position"])
        if matches_concept(n.get("label"), "hot")
    ]
    members, overflow = cap_members(hot_cands)
    sns.append(
        supernode(
            "hot",
            "hot",
            "donor",
            "hot_en",
            "operand",
            members,
            overflow,
            note="operand-swap donor; injected value = +1.5x the stored act",
        )
    )
    cold_cands = [
        member_entry(n, matched=f"top_logits:{matches_concept(n.get('label'), 'cold')!r}")
        for n in feature_nodes(graphs["hot_en"], final("hot_en"))
        if matches_concept(n.get("label"), "cold")
    ]
    members, overflow = cap_members(cold_cands)
    sns.append(supernode("say cold", "say cold", "readout", "hot_en", "final", members, overflow))

    # --- say large: multilingual (>=2 graphs) vs language-specific (exactly 1) --------
    for tag, names in (("", ant_names), ("raw ", raw_ant_names)):
        shared = shared_concept("large", names, final, min_graphs=2)
        members, overflow = cap_members(shared)
        sns.append(
            supernode(
                f"{tag}say large (multilingual)",
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
                    f"{tag}say large ({lg})",
                    f"say large ({lg})",
                    "readout",
                    name,
                    "final",
                    members_l,
                    overflow_l,
                )
            )

    # --- language detectors (graph-first; Fig B5 source/donor) ------------------------
    n_layers = 36
    detect_cands: dict[str, list[dict]] = {}
    for lg in LANGS:
        name = f"raw_antonym_{lg}"
        cands = []
        for n in feature_nodes(graphs[name], final(name)):
            if n["layer"] >= n_layers // 3:
                continue
            langs = example_langs(n.get("label"))
            if not langs:
                continue
            frac = langs.count(lg) / len(langs)
            if frac >= 0.7:
                cands.append(
                    member_entry(
                        n,
                        matched=f"examples:{langs.count(lg)}/{len(langs)} {lg}",
                        note="graph-first detector (early layer, language-pure examples)",
                    )
                )
        detect_cands[lg] = cands
    # language-unique: drop (L,f) appearing as a candidate for >=2 languages
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for lg in LANGS:
        for m in detect_cands[lg]:
            counts[(m["layer"], m["feature"])] += 1
    fallback_path = (
        artifacts_dir() / "paper_multilingual" / size / "raw_language_detection_supernodes.json"
    )
    fallback = json.loads(fallback_path.read_text()) if fallback_path.exists() else {}
    for lg in LANGS:
        uniq = [m for m in detect_cands[lg] if counts[(m["layer"], m["feature"])] == 1]
        members, overflow = cap_members(uniq)
        note = "language-swap source/donor (graph-first, raw open-quote graphs)"
        if not members:
            note += (
                " — EMPTY under graph-first selection (RECORDED FINDING: the pruned raw"
                " graphs contain NO early-layer nodes at the quote position at all, so"
                " the paper's detector supernodes cannot be graph nodes here). Members"
                " below are the OFF-GRAPH fallback: the v1 activation scan"
                " (early_language_detection_supernode on the raw prompts, committed in"
                f" {fallback_path.name}), marked source=off-graph — approve or reject at"
                " review."
            )
            fb = (fallback.get(lg) or [])[:MAX_MEMBERS]
            by_layer: dict[int, list[int]] = defaultdict(list)
            for L, i, _a in fb:
                by_layer[int(L)].append(int(i))
            labs: dict[tuple[int, int], dict] = {}
            for L, idxs in by_layer.items():
                for fi, lab_obj in load_feature_labels(repo_id, L, idxs).items():
                    d = lab_obj.to_dict()
                    d["examples"] = load_feature_examples(repo_id, L, idxs).get(fi, [])
                    labs[(L, fi)] = d
            for L, i, act in fb:
                lab_d = labs.get((int(L), int(i)))
                langs = example_langs(lab_d)
                members.append(
                    {
                        "layer": int(L),
                        "feature": int(i),
                        "position": "final",
                        "act": round(float(act), 4),
                        "influence": None,
                        "source": "off-graph-fallback",
                        "evidence": {
                            "matched": "v1 activation scan (early, language-unique); "
                            f"example langs {langs}",
                            "top_logits": (lab_d or {}).get("top_logits", [])[:8],
                            "example": example_snippet(lab_d),
                            "grid_class": None,
                        },
                        "review": "proposed",
                        "review_note": "not a pruned-graph node; off-graph activation pick",
                    }
                )
        sns.append(
            supernode(
                f"detect ({lg})",
                f"open-quote-in-{lg}",
                "source+donor",
                f"raw_antonym_{lg}",
                "final",
                members,
                overflow,
                note=note,
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
MANUAL_FLAGS = {
    ("small (multilingual)", 22, 113890): "top logits look unrelated (mistake/Sized/gest/"
    "intestine) and the example is an HTML tag list — doubtful 'small' feature",
    ("say cold", 29, 53713): "cold evidence weak (top: soft/heavy/冷/粗; example is about "
    "Pliny the Elder) — verify before keeping",
    ("say large (multilingual)", 31, 126548): "generic top logits (Very/Simple/Large/More) "
    "— check the activation examples before keeping",
}


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


def emit_review_htmls(root: Path, graphs: dict, docs: list[dict]) -> None:
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
                per_graph[gname].append(
                    {
                        "name": sn["name"],
                        "members": members,
                        "position": "final"
                        if pos == "final"
                        else None
                        if pos in (None, "operand")
                        else pos,
                    }
                )
    for gname, gspecs in per_graph.items():
        out_html = root / f"review_{gname}.html"
        render_graph_explorer_html(graphs[gname], out_html, title=f"review {gname}", groups=gspecs)
        print(f"  {out_html.name}: {len(gspecs)} proposed supernodes")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="4b")
    ap.add_argument("--check", action="store_true", help="validate committed files only")
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

    docs = []
    for task, builder in (("multilingual", build_multilingual), ("addition", build_addition)):
        doc = builder(args.size, root, manifest, graphs)
        apply_manual_flags(doc)
        errs = validate(doc, require_approved=False)
        if errs:
            raise SystemExit(f"{task}: validation failed:\n  " + "\n  ".join(errs))
        out = SUPERNODE_DIR / f"{task}_{args.size}.json"
        out.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
        docs.append(doc)
        print(f"{out} written:")
        for sn in doc["supernodes"]:
            flag = "" if sn["members"] else "   <-- EMPTY"
            print(
                f"   {sn['name']:34s} {len(sn['members'])} members"
                f" (+{len(sn.get('overflow', []))} overflow){flag}"
            )

    emit_review_htmls(root, graphs, docs)
    print(
        "\nNEXT: review the JSONs (and/or the review_*.html pages), edit members,"
        '\nset "approved": true — the notebooks refuse unapproved files.'
    )


if __name__ == "__main__":
    main()
