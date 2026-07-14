"""Stage 2 (CPU) of the supernode pipeline: seeds, review pages, and export ingest.

Scans ALL feature nodes of the pruned graphs persisted by ``build_supernode_inputs.py``
and SEEDS paper-aligned supernodes — multilingual by **label/example semantics**,
addition (v3) by **operand-grid geometry** (``addition_helper.classify_grid``) —
influence is recorded as evidence only, never used to select. The seeds only populate
the review pages: for BOTH tasks the authoritative selection is the human's explorer
"Export groups" JSONs, ingested via ``--from-exports`` (routed per page: addition
graphs carry ``task: "addition"`` in the manifest).

Outputs:
* ``artifacts/supernode_inputs/<size>/review_<graph>.html`` — reviewer explorer pages
  with the seed supernodes pre-loaded as groups. Addition pages are GRID-ENABLED:
  every feature node shows its operand plots (per-probe tabs, studied-pair mark,
  mod-10 guide) via ``attach_operand_grids``, plus "(overflow) ..." helper groups for
  candidates that lost the seed's size cut. KEEP THE CANONICAL GROUP NAMES
  (:func:`addition_names`) — the notebook drives supernodes by name.
* ``notebooks/supernodes/{multilingual_{chat,raw},addition}_<size>.json`` — written
  ONLY by the ingest paths, with hand-review provenance in ``review_log`` and
  ``approved: true``; the notebook loaders refuse anything unapproved.

Hard rules: supernodes are disjoint per selection; canonical addition groups must
arrive from their own page and the required ones must be non-empty — the script fails
loudly on violations.

Seeds + review pages:  uv run python notebooks/build_supernodes.py --size 4b
Ingest a hand review:  uv run python notebooks/build_supernodes.py --from-exports \\
                           notebooks/supernodes/exports/groups_<graph>.json ...
Check committed files: uv run python notebooks/build_supernodes.py --check
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
from multilingual_helper import OP_WORD

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


# Multilingual ranking (user decision 2026-07-12): EARLIEST layer first, activation as
# the tiebreak — early members leave the constrained-patching end-layer sweep room
# (l_max = max steered layer), where act-ranking systematically picks late layers.
def early_first(m: dict):
    return (-m["layer"], m["act"])


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
    """Seed proposal for the multilingual review pages.

    v3 (2026-07-12, user decisions): **chat and raw are separate selections** — every
    scan pools ONE format's graphs, identically-named groups merge per format at
    ingestion (two files), and members are ranked **earliest layer first**
    (:func:`early_first`) so the steered sets leave the constrained-patching end-layer
    sweep room. Every member records per-graph ``acts``/``positions`` so seeds can be
    materialized into export JSONs verbatim (``--materialize-seeds``).
    """
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

    def single_graph_concept(gname, *, position=None, matcher=None):
        """Best-act member per (L,f) on ONE graph, with acts/positions recorded.

        ``matcher(node) -> provenance str | None`` decides membership; ``position``
        restricts the scan (None = every position).
        """
        best: dict[tuple[int, int], dict] = {}
        for n in feature_nodes(graphs[gname], position):
            m = matcher(n)
            if not m:
                continue
            key = (n["layer"], n["feature_idx"])
            e = member_entry(n, matched=m)
            if key not in best or e["act"] > best[key]["act"]:
                e["acts"] = {gname: e["act"]}
                e["positions"] = {gname: n["position"]}
                best[key] = e
        return list(best.values())

    for fmt in ("chat", "raw"):
        p = "" if fmt == "chat" else "raw_"
        ant_names = [f"{p}antonym_{lg}" for lg in LANGS]
        syn_graph = f"{p}synonym_en"
        hot_graph = f"{p}hot_en"

        # --- operation swap: antonym (multilingual) source + synonym donor ------------
        # the antonym-operation supernode lives mid-graph (paper Fig B1: opposite+small
        # -> antonym), not only on the final token: scan every position, BOTH sides
        # (input-side = fires on "opposite"/"contraire"/反义; output-side = promotes).
        cands = shared_concept("opposite", ant_names, lambda _n: None, side="both")
        members, overflow = cap_members(cands, key=early_first)
        ant_keys = {(m["layer"], m["feature"]) for m in members + overflow}
        sns.append(
            supernode(
                "antonym (multilingual)",
                "antonym",
                "source",
                ";".join(ant_names),
                "member-positions",
                members,
                overflow,
                note="operation-swap source; steered -5x (m=-6) at each member's own"
                " node position per recipient graph",
            )
        )
        # paper-faithful donor: synonym OPERATION features — they fire ON the word
        # "synonym"/"synonymous" (input-side peaks) at the operation-word token of the
        # donor prompt; found only once the input side was searched (2026-07-11 review).
        op_best = single_graph_concept(
            syn_graph, matcher=lambda n: concept_match(n.get("label"), "synonym", "both")
        )
        members, overflow = cap_members(op_best, key=early_first)
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

        def _say_small_match(n):
            m = matches_concept(n.get("label"), "synonym_answer")
            if m and not matches_concept(n.get("label"), "large"):
                return f"top_logits:{m!r}"
            return None

        syn_cands = single_graph_concept(
            syn_graph, position=final(syn_graph), matcher=_say_small_match
        )
        members, overflow = cap_members(syn_cands, key=early_first)
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
                " prompt's final position.",
            )
        )

        # --- operand swap: small (multilingual) source + hot donor + say-cold readout -
        cands = shared_concept(
            "small",
            ant_names,
            lambda n: mg[n]["operand_position"],
            side="both",
        )
        members, overflow = cap_members(cands, key=early_first)
        small_keys = {(m["layer"], m["feature"]) for m in members + overflow}
        sns.append(
            supernode(
                "small (multilingual)",
                "small",
                "source",
                ";".join(ant_names),
                "operand",
                members,
                overflow,
                note="operand-swap source; steered -0.5x (m=-1.5). Input-side matches"
                " (peak example token = small/petit/...) are the paper's operand-feature"
                " analogue; output-side matches at the operand token are say-small-ish",
            )
        )
        cands = single_graph_concept(
            hot_graph,
            position=mg[hot_graph]["operand_position"],
            matcher=lambda n: concept_match(n.get("label"), "hot", "both"),
        )
        members, overflow = cap_members(cands, key=early_first)
        sns.append(
            supernode(
                "hot (multilingual)",
                "hot",
                "donor",
                hot_graph,
                "operand",
                members,
                overflow,
                note="operand-swap donor; injected value = +1.5x the stored act",
            )
        )
        cands = single_graph_concept(
            hot_graph,
            position=final(hot_graph),
            matcher=lambda n: concept_match(n.get("label"), "cold", "output"),
        )
        members, overflow = cap_members(cands, key=early_first)
        sns.append(
            supernode(
                "say cold (multilingual)",
                "say cold",
                "readout",
                hot_graph,
                "final",
                members,
                overflow,
            )
        )

        # --- say large: multilingual (>=2 graphs) vs language-specific (exactly 1) ----
        shared = shared_concept("large", ant_names, final, min_graphs=2)
        members, overflow = cap_members(shared, key=early_first)
        shared_keys = {(m["layer"], m["feature"]) for m in members + overflow}
        lang_of: dict[tuple[int, int], set[str]] = defaultdict(set)
        for lg in LANGS:
            gname = ant_names[LANGS.index(lg)]
            for n in feature_nodes(graphs[gname], final(gname)):
                key = (n["layer"], n["feature_idx"])
                if matches_concept(n.get("label"), "large") and key not in shared_keys:
                    lang_of[key].add(lg)
        sns.append(
            supernode(
                "say large (multilingual)",
                "say large",
                "readout",
                ";".join(ant_names),
                "final",
                members,
                overflow,
            )
        )
        for lg in LANGS:
            gname = ant_names[LANGS.index(lg)]

            def _say_large_lang(n, _lg=lg, _shared=shared_keys, _lang_of=lang_of):
                key = (n["layer"], n["feature_idx"])
                m = matches_concept(n.get("label"), "large")
                if not m or key in _shared or _lang_of[key] != {_lg}:
                    return None
                return f"top_logits:{m!r}"

            only = single_graph_concept(gname, position=final(gname), matcher=_say_large_lang)
            for entry in only:
                # script sanity: a {lang}-specific say-large whose only match is a
                # different script (e.g. a CJK token on the FR graph) is suspect
                tok = entry["evidence"]["matched"].split("top_logits:", 1)[-1]
                tok_lang = text_lang(tok)
                if (lg == "zh") != (tok_lang == "zh"):
                    entry["review_note"] = (
                        f"matched token {tok} is {tok_lang}-script on the {lg}-specific"
                        " supernode — likely junk/mixed feature, verify the example"
                    )
            members_l, overflow_l = cap_members(only, key=early_first)
            sns.append(
                supernode(
                    f"say large ({lg})",
                    f"say large ({lg})",
                    "readout",
                    gname,
                    "final",
                    members_l,
                    overflow_l,
                )
            )

        # --- language detectors (raw pages only; Fig B5 source/donor) -----------------
        # The paper's detectors sit at the final open-quote token, early layers.
        # Qwen3's 0.95 graphs have NO layer<12 nodes at that position, so the seed
        # keeps only the semantic filter (language-pure examples), earliest first.
        quote_keys: set[tuple[int, int]] = set()
        if fmt == "raw":
            detect_cands: dict[str, list[dict]] = {}
            for lg in LANGS:
                gname = f"raw_antonym_{lg}"

                def _detector(n, _lg=lg):
                    langs = example_langs(n.get("label"))
                    if langs and langs.count(_lg) / len(langs) >= 0.7:
                        return f"examples:{langs.count(_lg)}/{len(langs)} {_lg}"
                    return None

                detect_cands[lg] = single_graph_concept(
                    gname, position=final(gname), matcher=_detector
                )
            counts: dict[tuple[int, int], int] = defaultdict(int)
            for lg in LANGS:
                for m in detect_cands[lg]:
                    counts[(m["layer"], m["feature"])] += 1
            for lg in LANGS:
                uniq = [m for m in detect_cands[lg] if counts[(m["layer"], m["feature"])] == 1]
                members, overflow = cap_members(uniq, key=early_first)
                quote_keys |= {(m["layer"], m["feature"]) for m in members + overflow}
                sns.append(
                    supernode(
                        f"quote ({lg})",
                        f"open-quote-in-{lg}",
                        "source+donor",
                        f"raw_antonym_{lg}",
                        "final",
                        members,
                        overflow,
                        note="language-swap source/donor (paper: 'quote"
                        " (lang-specific)'; language-pure examples, earliest layers"
                        " first — Qwen3's quote-position nodes all sit above L12,"
                        " unlike the paper's early detectors).",
                    )
                )

        # --- opposite (lang-specific): the operation WORD's own features --------------
        op_cands: dict[str, dict[tuple[int, int], dict]] = {}
        for lg in LANGS:
            gname = ant_names[LANGS.index(lg)]
            positions = op_word_positions(graphs[gname], OP_WORD[lg])

            def _op_word(n, _pos=positions, _keys=ant_keys | quote_keys | small_keys, _lg=lg):
                # disjointness: features already seeded into antonym/quote/small stay
                # there (the paper's quote-features track language via other words too)
                key = (n["layer"], n["feature_idx"])
                if n["position"] not in _pos or key in _keys:
                    return None
                return f"at operation-word token {OP_WORD[_lg]!r}"

            op_cands[lg] = {
                (m["layer"], m["feature"]): m for m in single_graph_concept(gname, matcher=_op_word)
            }
        for lg in LANGS:
            # language-unique only: a feature on the operation word of >=2 languages is
            # not lang-specific (and not auto-promoted to antonym — admit ambiguity)
            uniq = [
                e for key, e in op_cands[lg].items() if sum(key in op_cands[o] for o in LANGS) == 1
            ]
            members, overflow = cap_members(uniq, key=early_first)
            sns.append(
                supernode(
                    f"opposite ({lg})",
                    f"opposite ({lg})",
                    "input",
                    ant_names[LANGS.index(lg)],
                    "member-positions",
                    members,
                    overflow,
                    note="language-specific operation-word features (upstream evidence;"
                    " not steered in the paper's three swaps).",
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
# Addition v3 (clean-room): operand grids ARE the evidence; the hand review in
# the grid-enabled explorer is the selection. The scan below only SEEDS the
# review pages — the authoritative file is written by --from-exports.
# ---------------------------------------------------------------------------


class GridCtx:
    """``grids.npz`` index + memoized classifier reports.

    Shared by the seed scan, the review-page grid attachment, and the export ingest so
    every consumer sees the same (probe, layer, feature) -> grid/class mapping. Probe
    arrays decompress lazily (once each).
    """

    PROBES = ("final", "ones", "peak")

    def __init__(self, root: Path):
        self._z = np.load(root / "grids.npz")
        self.index: dict[str, dict[tuple[int, int], int]] = {
            probe: {
                (int(layer), int(feat)): i
                for i, (layer, feat) in enumerate(self._z[f"{probe}_features"])
            }
            for probe in self.PROBES
            if f"{probe}_features" in self._z
        }
        self._arrays: dict[str, np.ndarray] = {}
        self._reports: dict[tuple[str, int, int], tuple[str, dict] | None] = {}

    def grid(self, probe: str, layer: int, feat: int) -> np.ndarray | None:
        i = self.index.get(probe, {}).get((layer, feat))
        if i is None:
            return None
        if probe not in self._arrays:
            self._arrays[probe] = self._z[f"{probe}_grids"]
        return self._arrays[probe][i]

    def report(self, probe: str, layer: int, feat: int) -> tuple[str, dict] | None:
        key = (probe, layer, feat)
        if key not in self._reports:
            g = self.grid(probe, layer, feat)
            self._reports[key] = None if g is None else A.classify_grid(g)
        return self._reports[key]


def load_grid_ctx(root: Path) -> GridCtx | None:
    return GridCtx(root) if (root / "grids.npz").exists() else None


def addition_names(manifest: dict) -> dict[str, dict]:
    """The canonical v3 supernode names -> {role, graph, position} (pair-parametric).

    These are the group names the review pages seed and the notebook drives; the
    ingest hard-checks the required ones, so REVIEWERS MUST NOT RENAME THEM (adding
    custom groups is fine — they ingest with role "custom").
    """
    add = manifest["addition"]
    (a0, b0), (da_, db_) = add["pair"], add["donor_pair"]
    ra, rb, s0 = a0 % 10, b0 % 10, a0 + b0
    dra, drb = da_ % 10, db_ % 10
    dp = manifest["graphs"]["ones"]["digit_positions"]

    def spec(role, graph, position, required=False, note=""):
        return {
            "role": role,
            "graph": graph,
            "position": position,
            "required": required,
            "note": note,
        }

    return {
        f"input _{ra}": spec("source", "ones", dp["a_digits"][-1], required=True),
        f"input _{rb}": spec("source", "ones", dp["b_digits"][-1], required=True),
        f"input {a0} (exact)": spec("annotation", "ones", None),
        f"input {b0} (exact)": spec("annotation", "ones", None),
        f"magnitude ~{a0}": spec("source", "first", None),
        f"magnitude ~{b0}": spec("source", "first", None),
        f"add ~{b0} (function)": spec("annotation", "first", "final"),
        f"magnitude lookup (~{a0}+~{b0})": spec("annotation", "first", "final"),
        f"sum ~{s0} (low precision)": spec("readout", "first", "final", required=True),
        f"lookup (_{ra}+_{rb})": spec("source", "ones", "final", required=True),
        f"sum = _{s0 % 10}": spec("readout", "ones", "final", required=True),
        f"lookup (_{dra}+_{drb}) donors": spec("donor", "donor", "final", required=True),
        f"reuse lookups (_{ra}+_{rb})": spec("annotation", "reuse", "final"),
    }


def _addition_graph_names(manifest: dict) -> set[str]:
    return {g for g, e in manifest["graphs"].items() if e.get("task") == "addition"}


def build_addition_seed(size: str, root: Path, manifest: dict, graphs: dict, ctx: GridCtx) -> dict:
    """Seed proposal for the addition review pages, from operand-grid geometry alone.

    Pure scan output (no baked-in review decisions): each candidate's grid class under
    the group's probe must match the paper family, ranked earliest-layer-first (leaves
    the constrained-patching sweep room), capped at MAX_MEMBERS with the remainder kept
    as ``overflow`` (surfaced on the review pages as "(overflow) ..." groups).
    """
    mg = manifest["graphs"]
    add = manifest["addition"]
    (a0, b0), (da_, db_) = add["pair"], add["donor_pair"]
    ra, rb, s0 = a0 % 10, b0 % 10, a0 + b0
    dra, drb = da_ % 10, db_ % 10
    dp = mg["ones"]["digit_positions"]
    names = addition_names(manifest)
    EARLY = 12  # input-identity features live in the first third of the 36 layers

    def collect(gname, positions, probe, pred, *, max_layer=None):
        best: dict[tuple[int, int], dict] = {}
        for n in feature_nodes(graphs[gname]):
            if positions is not None and n["position"] not in positions:
                continue
            if max_layer is not None and n["layer"] >= max_layer:
                continue
            rep = ctx.report(probe, n["layer"], n["feature_idx"])
            if rep is None:
                continue
            cls, stats = rep
            if not pred(cls, stats):
                continue
            key = (n["layer"], n["feature_idx"])
            e = member_entry(n, matched=f"grid[{probe}]:{cls}", grid_class=cls)
            if key not in best or e["act"] > best[key]["act"]:
                best[key] = e
        return list(best.values())

    def near(stats_key, center, tol):
        def pred(cls, stats):
            fam = {"band_a": "band-a", "band_b": "band-b", "sum_band": "sum-band"}[stats_key]
            return cls.startswith(fam) and abs(stats[stats_key][0] + 7 - center) <= tol

        return pred

    sns: list[dict] = []

    def add_sn(name, cands, note=""):
        spec = names[name]
        members, overflow = cap_members(cands, key=early_first)
        sns.append(
            supernode(
                name,
                name,
                spec["role"],
                spec["graph"],
                spec["position"],
                members,
                overflow,
                note=note,
            )
        )

    fin = {g: mg[g]["final_position"] for g in ("first", "ones", "donor", "reuse") if g in mg}

    # inputs: the operand's ones-digit identity, read at its digit tokens (peak probe)
    def residue_pred(res_key, residue):
        def pred(cls, stats):
            r, share = stats.get(res_key, [None, 0.0])[:2]
            return r == residue and share > 0.4

        return pred

    add_sn(
        f"input _{ra}",
        collect("ones", set(dp["a_digits"]), "peak", residue_pred("mod10_a", ra), max_layer=EARLY),
        note="fires when a's ones digit is the studied residue (mod10-a mass > 0.4)",
    )
    add_sn(
        f"input _{rb}",
        collect("ones", set(dp["b_digits"]), "peak", residue_pred("mod10_b", rb), max_layer=EARLY),
        note="fires when b's ones digit is the studied residue (mod10-b mass > 0.4)",
    )

    def exact_pred(value, side):
        def pred(cls, stats):
            r_star, c_star, rs, cs = stats.get("cross", [None, None, 0.0, 0.0])
            return (
                (r_star == value and rs > 0.28) if side == "a" else (c_star == value and cs > 0.28)
            )

        return pred

    add_sn(
        f"input {a0} (exact)",
        collect("ones", set(dp["a_digits"]), "peak", exact_pred(a0, "a"), max_layer=EARLY),
        note="exact-value identity (single-row mass > 0.28 at a=pair)",
    )
    add_sn(
        f"input {b0} (exact)",
        collect("ones", set(dp["b_digits"]), "peak", exact_pred(b0, "b"), max_layer=EARLY),
        note="exact-value identity (single-column mass > 0.28 at b=pair)",
    )

    # magnitude path (the FIRST-digit graph): operand-magnitude bands at the digits,
    # add-function bands + the low-precision joint region + the sum band at '='
    add_sn(
        f"magnitude ~{a0}",
        collect(
            "first", set(mg["first"]["digit_positions"]["a_digits"]), "peak", near("band_a", a0, 6)
        ),
    )
    add_sn(
        f"magnitude ~{b0}",
        collect(
            "first", set(mg["first"]["digit_positions"]["b_digits"]), "peak", near("band_b", b0, 6)
        ),
    )
    add_sn(
        f"add ~{b0} (function)",
        collect("first", {fin["first"]}, "final", near("band_b", b0, 6)),
        note="the paper's 'add something near-b' function features, read at '='",
    )
    add_sn(
        f"magnitude lookup (~{a0}+~{b0})",
        collect(
            "first",
            {fin["first"]},
            "final",
            lambda cls, st: (
                cls.startswith("region")
                and abs(st["peak"][0] - a0) <= 8
                and abs(st["peak"][1] - b0) <= 8
            ),
        ),
    )
    add_sn(
        f"sum ~{s0} (low precision)",
        collect("first", {fin["first"]}, "final", near("sum_band", s0, 8)),
    )

    # high-precision modular path (the ONES graph at its teacher-forced moment)
    add_sn(
        f"lookup (_{ra}+_{rb})",
        collect(
            "ones", {fin["ones"]}, "ones", lambda cls, st: cls == f"lookup(a%10={ra},b%10={rb})"
        ),
    )
    add_sn(
        f"sum = _{s0 % 10}",
        collect("ones", {fin["ones"]}, "ones", lambda cls, st: cls == f"mod10-sum(r{s0 % 10})"),
    )

    # the donor problem's lookup features (the paper's substitution source)
    add_sn(
        f"lookup (_{dra}+_{drb}) donors",
        collect(
            "donor", {fin["donor"]}, "ones", lambda cls, st: cls == f"lookup(a%10={dra},b%10={drb})"
        ),
    )

    # the same studied-pair lookup class appearing on the prose reuse page
    if "reuse" in fin:
        add_sn(
            f"reuse lookups (_{ra}+_{rb})",
            collect(
                "reuse",
                {fin["reuse"]},
                "ones",
                lambda cls, st: cls == f"lookup(a%10={ra},b%10={rb})",
            ),
            note="calc lookup class firing at the citation prompt's ones moment",
        )

    return {
        "task": "addition",
        "size": size,
        "built_from": f"{root}@{manifest.get('git', '?')}",
        "max_members": MAX_MEMBERS,
        "supernodes": sns,
        "approved": False,
    }


def attach_operand_grids(gd: dict, gname: str, entry: dict, ctx: GridCtx) -> dict:
    """A copy of graph dict *gd* with per-feature operand grids wired for the explorer.

    Every feature node gets ``label.operand_grids`` refs (position-appropriate probe
    first) into a page-level ``operand_grid_store`` of :mod:`grid_codec`-encoded grids;
    the studied pair is marked on calc/reuse pages. The input *gd* is not mutated.
    """
    from llm_circuits.circuits.grid_codec import encode_grid_u8

    pair = entry.get("pair")
    fin = entry.get("final_position")
    target = str(entry.get("target", ""))
    final_probe = "ones" if target in ("ones", "reuse-ones") else "final"
    stat_keys = (
        "frac_on",
        "lattice_cell",
        "mod10_sum",
        "band_a",
        "band_b",
        "sum_band",
        "cross",
        "peak",
    )

    store: dict[str, dict] = {}
    nodes = []
    for nd in gd["nodes"]:
        nd = dict(nd)
        if nd["node_type"] == "feature":
            layer, feat = nd["layer"], nd["feature_idx"]
            primary = final_probe if nd["position"] == fin else "peak"
            refs = []
            for probe in (primary, *[p for p in GridCtx.PROBES if p != primary]):
                rep = ctx.report(probe, layer, feat)
                if rep is None:
                    continue
                cls, stats = rep
                key = f"{probe}:L{layer}f{feat}"
                if key not in store:
                    store[key] = encode_grid_u8(ctx.grid(probe, layer, feat))
                refs.append(
                    {
                        "probe": probe,
                        "key": key,
                        "cls": cls,
                        "mark": list(pair) if pair else None,
                        "pair_frac": (
                            round(A.on_pair_fraction(ctx.grid(probe, layer, feat), tuple(pair)), 3)
                            if pair
                            else None
                        ),
                        "stats": {k: stats[k] for k in stat_keys if k in stats},
                    }
                )
            if refs:
                label = dict(nd.get("label") or {})
                label["operand_grids"] = refs
                nd["label"] = label
        nodes.append(nd)
    out = dict(gd)
    out["nodes"] = nodes
    out["operand_grid_store"] = store
    return out


def ingest_addition_exports(
    size, files, root, manifest, graphs, ctx: GridCtx, review_log: str | None = None
) -> None:
    """Write ``addition_<size>.json`` FROM the review pages' 'Export groups' JSONs.

    v3: the hand review in the grid-enabled explorer IS the selection. Groups keep the
    canonical names (:func:`addition_names` — renaming a required group is a hard
    error because the notebook drives supernodes by name); unknown names ingest with
    role "custom" and need wiring before they steer. Addition supernodes are per-graph
    selections (no cross-page merge); "(overflow) ..." helper groups are skipped.
    Evidence (grid class, on-pair fraction) is auto-filled from ``grids.npz``; the
    written file must pass :func:`validate` and ``addition_helper.load_supernodes``.
    """
    names = addition_names(manifest)
    add = manifest["addition"]
    pair, donor_pair = tuple(add["pair"]), tuple(add["donor_pair"])
    sns: dict[str, dict] = {}
    src_files: list[str] = []

    for f in files:
        exp = json.loads(Path(f).read_text())
        gname = str(exp.get("example", "")).strip()
        if manifest["graphs"].get(gname, {}).get("task") != "addition" or gname not in graphs:
            raise SystemExit(
                f"{f}: example label {gname!r} is not a dumped addition graph — re-export"
                f" from the CURRENT review pages (addition: {sorted(_addition_graph_names(manifest))})"
            )
        src_files.append(Path(f).name)
        entry = manifest["graphs"][gname]
        final_probe = "ones" if str(entry.get("target", "")) in ("ones", "reuse-ones") else "final"
        nodes_at = {
            (n["layer"], n["feature_idx"], n["position"]): n for n in feature_nodes(graphs[gname])
        }
        claimed: dict[tuple[int, int], str] = {}
        for grp in exp.get("groups", []):
            name = str(grp["name"]).strip()
            if name.startswith("(overflow)"):
                print(f"  [skip] {gname}: helper group {name!r}")
                continue
            spec = names.get(name)
            if spec and spec["graph"] != gname:
                raise SystemExit(
                    f"{f}: canonical group {name!r} belongs on the {spec['graph']!r} page,"
                    f" not {gname!r} — regroup there and re-export"
                )
            if name in sns:
                raise SystemExit(
                    f"{f}: group {name!r} appears on two addition pages — addition"
                    " supernodes are per-graph selections (no cross-page merge)"
                )
            members: list[dict] = []
            for ndref in grp.get("nodes", []):
                key = (int(ndref["layer"]), int(ndref["feature_idx"]))
                if key in claimed and claimed[key] != name:
                    raise SystemExit(
                        f"{f}: L{key[0]}f{key[1]} is in both {claimed[key]!r} and {name!r}"
                        " — supernodes are disjoint on a page; fix in the UI and re-export"
                    )
                claimed[key] = name
                node = nodes_at.get((key[0], key[1], ndref.get("position")))
                probe = (
                    final_probe if ndref.get("position") == entry.get("final_position") else "peak"
                )
                rep = ctx.report(probe, key[0], key[1])
                cls = rep[0] if rep else None
                if node is not None:
                    m = member_entry(node, matched="manual (explorer export)", grid_class=cls)
                else:
                    m = {
                        "layer": key[0],
                        "feature": key[1],
                        "position": ndref.get("position"),
                        "act": 0.0,
                        "influence": None,
                        "evidence": {
                            "matched": "manual (explorer export; not found in dump at that"
                            " position)",
                            "top_logits": [],
                            "example": "",
                            "grid_class": cls,
                        },
                    }
                m["source"] = "explorer-export"
                m["review"] = "approved"
                grid = ctx.grid(probe, key[0], key[1])
                if grid is not None:
                    ref_pair = donor_pair if (spec or {}).get("role") == "donor" else pair
                    m["evidence"]["on_pair_frac_of_max"] = round(
                        A.on_pair_fraction(grid, ref_pair), 3
                    )
                members.append(m)
            members.sort(key=lambda m: -m["act"])
            positions = {m.get("position") for m in members}
            if spec:
                position = spec["position"]
            elif positions == {entry.get("final_position")}:
                position = "final"
            elif len(positions) == 1:
                position = next(iter(positions))
            else:
                position = None
            sns[name] = {
                "name": name,
                "paper_name": name,
                "role": spec["role"] if spec else "custom",
                "graph": gname,
                "position": position,
                "members": members,
            }
            if not spec:
                sns[name]["note"] = "custom group from the hand review — wire before steering"

    missing_required = [
        n for n, spec in names.items() if spec["required"] and not sns.get(n, {}).get("members")
    ]
    if missing_required:
        raise SystemExit(
            "required supernodes empty or missing after ingest: "
            + ", ".join(repr(n) for n in missing_required)
            + " — the notebook cannot run without them (do not rename canonical groups)"
        )
    mag_names = [n for n in names if n.startswith("magnitude ~")]
    if not any(sns.get(n, {}).get("members") for n in mag_names):
        raise SystemExit(
            f"both magnitude groups ({', '.join(mag_names)}) are empty — keep at least one"
        )
    for n, spec in names.items():  # keep measured-negative annotations visible
        if n not in sns:
            sns[n] = {
                "name": n,
                "paper_name": n,
                "role": spec["role"],
                "graph": spec["graph"],
                "position": spec["position"],
                "members": [],
                "note": "absent after hand review (no group exported)",
            }

    ordered = [sns[n] for n in names if n in sns]
    ordered += [sn for n, sn in sns.items() if n not in names]
    biggest = max((len(sn["members"]) for sn in ordered), default=0)
    doc = {
        "task": "addition",
        "size": size,
        "built_from": f"{root}@{manifest.get('git', '?')}",
        "selection": "explorer-export (grid-enabled review pages)",
        "source_files": sorted(src_files),
        "max_members": max(MAX_MEMBERS, biggest),
        "supernodes": ordered,
        "approved": True,
        "review_log": review_log
        or "selected and reviewed by hand in the grid-enabled explorer (Export groups);"
        " ingested by build_supernodes.py --from-exports",
    }
    errs = validate(doc, require_approved=True)
    if errs:
        raise SystemExit("addition: validation failed:\n  " + "\n  ".join(errs))
    SUPERNODE_DIR.mkdir(parents=True, exist_ok=True)
    out = SUPERNODE_DIR / f"addition_{size}.json"
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    A.load_supernodes(out)  # the notebook loader is the final gate
    print(f"{out} written:")
    for sn in doc["supernodes"]:
        flag = "" if sn["members"] else "   <-- empty (documented absence)"
        print(f"   {sn['name']:34s} {len(sn['members'])} members{flag}")


# ---------------------------------------------------------------------------
# Disjointness + validation + reviewer HTMLs
# ---------------------------------------------------------------------------


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


def emit_review_htmls(
    root: Path, manifest: dict, graphs: dict, docs: list[dict], *, ctx: GridCtx | None = None
) -> None:
    per_graph: dict[str, list[dict]] = defaultdict(list)
    for doc in docs:
        is_addition = doc.get("task") == "addition"
        for sn in doc["supernodes"]:
            for gname in str(sn["graph"]).split(";"):
                if gname not in graphs:
                    continue
                pos = sn["position"]
                # Only "final" and integer positions restrict resolution; markers like
                # "operand"/"member-positions" mean the (L,f) members pin the nodes.
                pos = pos if pos == "final" or isinstance(pos, int) else None
                members = [(m["layer"], m["feature"]) for m in sn["members"]]
                if members:
                    per_graph[gname].append(
                        {"name": sn["name"], "members": members, "position": pos}
                    )
                if is_addition and sn.get("overflow"):
                    # candidates that lost the seed's size cut — exactly what the hand
                    # review should reconsider; ingest skips "(overflow) ..." groups
                    per_graph[gname].append(
                        {
                            "name": f"(overflow) {sn['name']}",
                            "members": [(m["layer"], m["feature"]) for m in sn["overflow"]],
                            "position": pos,
                        }
                    )
    for gname in _addition_graph_names(manifest):
        if gname in graphs:
            per_graph.setdefault(gname, [])  # addition pages emit even when unseeded
    for gname, gspecs in sorted(per_graph.items()):
        # Filename convention: review_chat_* / review_raw_* for the multilingual pairs
        # (the manifest's "raw" flag exists only for those); addition stays review_*.
        # The DUMP name (= export "example" label) is unprefixed either way.
        entry = manifest["graphs"].get(gname, {})
        raw_flag = entry.get("raw")
        stem = f"chat_{gname}" if raw_flag is False else gname
        gd = graphs[gname]
        if ctx is not None and entry.get("task") == "addition":
            gd = attach_operand_grids(gd, gname, entry, ctx)  # the grid-enabled pages
        out_html = root / f"review_{stem}.html"
        render_graph_explorer_html(
            gd,
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


def ingest_exports(size, files, root, manifest, graphs, review_log: str | None = None) -> None:
    """Write the multilingual supernode files FROM the 'Export groups' JSONs.

    v3: **chat and raw are separate selections** — each export's page format (the
    manifest's ``raw`` flag for its ``example`` graph) routes its groups into
    ``multilingual_chat_<size>.json`` or ``multilingual_raw_<size>.json``. Within a
    format, same-named groups across pages merge into one multi-graph supernode
    (per-graph ``acts``/``positions`` recorded — donors take value = mult x their own
    graph's act; sources steer at each member's node position per recipient) and
    disjointness by (layer, feature) is enforced; the same feature MAY appear in both
    formats' files (they are independent selections). Evidence is auto-filled from the
    graph dumps. ``review_log`` records the selection provenance (default: hand review
    in the explorer).
    """
    merged: dict[str, dict[str, dict]] = {"chat": {}, "raw": {}}
    files_by_fmt: dict[str, list[str]] = {"chat": [], "raw": []}
    for f in files:
        exp = json.loads(Path(f).read_text())
        gname = str(exp.get("example", "")).strip()
        if gname not in graphs:
            raise SystemExit(
                f"{f}: example label {gname!r} is not a dumped graph — re-export from"
                f" the CURRENT review pages (known: {sorted(graphs)})"
            )
        fmt = "raw" if manifest["graphs"][gname].get("raw") else "chat"
        files_by_fmt[fmt].append(str(Path(f).name))
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
            sn = merged[fmt].setdefault(
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

    for fmt, fmt_merged in merged.items():
        if not fmt_merged:
            continue
        # Paper supernodes are disjoint FEATURE sets within a selection: the same
        # (layer, feature) may not appear in two groups anywhere across this format's
        # pages (per-page overlap is caught above; this catches e.g. 'antonym' on the
        # EN page vs 'opposite (fr)' on the FR page).
        owner: dict[tuple[int, int], str] = {}
        for sn in fmt_merged.values():
            for key in sn["members"]:
                if key in owner and owner[key] != sn["name"]:
                    raise SystemExit(
                        f"[{fmt}] feature L{key[0]}f{key[1]} is in both {owner[key]!r}"
                        f" and {sn['name']!r} (possibly on different pages) —"
                        " supernodes are disjoint feature sets; fix and re-export"
                    )
                owner[key] = sn["name"]

        supernodes = []
        for sn in fmt_merged.values():
            members = sorted(sn["members"].values(), key=lambda m: -max(m["acts"].values() or [0]))
            note = "selection ingested verbatim from the committed export JSONs"
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
            "format": fmt,
            "built_from": f"{root}@{manifest.get('git', '?')}",
            "selection": "explorer-export",
            "source_files": files_by_fmt[fmt],
            # per-PAGE groups are capped at MAX_MEMBERS by the seeds; the union across
            # a format's pages may exceed it — the export record is authoritative
            "max_members": max((len(s["members"]) for s in supernodes), default=MAX_MEMBERS),
            "supernodes": supernodes,
            "approved": True,
            "review_log": review_log
            or "selected and reviewed by hand in the explorer (Export groups);"
            " ingested by build_supernodes.py --from-exports",
        }
        errs = validate(doc, require_approved=True)
        if errs:
            raise SystemExit(
                f"ingest [{fmt}]: validation failed (fix the groups and re-export):\n  "
                + "\n  ".join(errs)
            )
        out = SUPERNODE_DIR / f"multilingual_{fmt}_{size}.json"
        out.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
        print(f"{out} written from {len(files_by_fmt[fmt])} exports:")
        for sn in supernodes:
            print(
                f"   {sn['name']:34s} {len(sn['members'])} members  [{sn['role']}]  {sn['graph']}"
            )


def materialize_seed_exports(seed: dict, graphs: dict) -> list[Path]:
    """Write ``exports/groups_<graph>.json`` for every multilingual page FROM the seed
    proposal — the "accept the seeds as-is" flow, made explicit and reproducible.

    Each seed member carries per-graph ``positions``; a page's export lists every seed
    group with the members present on that page, at their recorded node positions
    (the same schema the explorer's "Export groups" button produces). Returns the
    written paths (ingest them with :func:`ingest_exports`).
    """
    export_dir = SUPERNODE_DIR / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    pages: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for sn in seed["supernodes"]:
        for m in sn["members"]:
            for gname, pos in (m.get("positions") or {}).items():
                if gname not in graphs:
                    continue
                pages[gname][sn["name"]].append(
                    {"layer": m["layer"], "feature_idx": m["feature"], "position": pos}
                )
    written = []
    for gname in sorted(pages):
        doc = {
            "example": gname,
            "groups": [
                {"name": name, "nodes": nodes} for name, nodes in pages[gname].items() if nodes
            ],
        }
        out = export_dir / f"groups_{gname}.json"
        out.write_text(json.dumps(doc, indent=1, ensure_ascii=False))
        written.append(out)
        print(f"  {out.name}: {len(doc['groups'])} seeded groups materialized")
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="4b")
    ap.add_argument("--check", action="store_true", help="validate committed files only")
    ap.add_argument(
        "--from-exports",
        nargs="+",
        metavar="EXPORT_JSON",
        help="ingest 'Export groups' JSONs (one per graph) and write the multilingual"
        " supernode files from them (chat/raw split by page format) — the ONLY"
        " multilingual selection source; evidence is auto-filled from the graph dumps",
    )
    ap.add_argument(
        "--materialize-seeds",
        action="store_true",
        help="write exports/groups_<graph>.json FROM the seed proposal and ingest them"
        " (the accept-seeds-as-is flow, recorded as such in the review_log)",
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
    ctx = load_grid_ctx(root)
    add_graphs = _addition_graph_names(manifest)
    ml_present = any(
        g in graphs for g in manifest["graphs"] if manifest["graphs"][g].get("task") != "addition"
    )
    SUPERNODE_DIR.mkdir(exist_ok=True)

    if args.from_exports:
        # Route each export by its page's manifest entry (addition graphs carry
        # task="addition" — checked FIRST, so raw-format addition prompts can never
        # fall into the multilingual raw router).
        add_files, ml_files = [], []
        for f in args.from_exports:
            gname = str(json.loads(Path(f).read_text()).get("example", "")).strip()
            is_add = manifest["graphs"].get(gname, {}).get("task") == "addition"
            (add_files if is_add else ml_files).append(f)
        if add_files:
            if ctx is None:
                raise SystemExit("grids.npz missing — the addition ingest needs grid evidence")
            ingest_addition_exports(args.size, add_files, root, manifest, graphs, ctx)
        if ml_files:
            ingest_exports(args.size, ml_files, root, manifest, graphs)
        return

    if args.materialize_seeds:
        seed = build_multilingual(args.size, root, manifest, graphs)
        files = materialize_seed_exports(seed, graphs)
        ingest_exports(
            args.size,
            files,
            root,
            manifest,
            graphs,
            review_log="seeds materialized and approved by user instruction"
            " (2026-07-12): earliest-first ranking, chat/raw selections separated;"
            " ingested by build_supernodes.py --materialize-seeds",
        )
        return

    # Default: SEEDS ONLY, for both tasks — nothing writes the reviewed files here.
    # The authoritative selections are the human's explorer "Export groups" JSONs,
    # ingested via --from-exports (v3: addition included; the old delegated-review
    # scan-write path is retired).
    docs = []
    if add_graphs & set(graphs):
        if ctx is None:
            raise SystemExit("grids.npz missing — run build_supernode_inputs.py first")
        seed_add = build_addition_seed(args.size, root, manifest, graphs, ctx)
        docs.append(seed_add)
        print("addition: grid-scan seeds for the review pages (no file written)")
        for sn in seed_add["supernodes"]:
            flag = "" if sn["members"] else "   <-- empty seed"
            print(f"   {sn['name']:34s} {len(sn['members'])} seeded{flag}")

    if ml_present:
        seed = build_multilingual(args.size, root, manifest, graphs)
        docs.append(seed)
        print("\nmultilingual: scan used as review-page seeds only (no file written)")
        for sn in seed["supernodes"]:
            flag = "" if sn["members"] else "   <-- empty seed"
            print(f"   {sn['name']:34s} {len(sn['members'])} seeded{flag}")
    else:
        print("\nmultilingual: no dumps on this machine — skipping its seeds/pages")

    emit_review_htmls(root, manifest, graphs, docs, ctx=ctx)
    print(
        "\nNEXT: review the review_*.html pages (addition pages show each feature's"
        "\noperand grid — keep the canonical group names), Export groups per page,"
        "\nthen: build_supernodes.py --from-exports <files...>"
    )


if __name__ == "__main__":
    main()
