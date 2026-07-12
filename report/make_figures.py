"""Report figures for Section 3 (multilingual): graphs, interventions, overlap.

Run:  uv run python report/make_figures.py   (reads committed artifacts, writes
report/figures/*.pdf; mplfonts supplies the CJK glyphs).

All content is drawn from the committed artifacts — nothing is schematic except the
box layouts; edge widths, % readouts, curves, and crossovers are measured values.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts/paper_multilingual/4b"
SNI = REPO / "artifacts/supernode_inputs/4b"
OUT = REPO / "report/figures"
OUT.mkdir(exist_ok=True)

# CJK glyphs (大 / 冷 / 小) via the mplfonts-bundled Noto Sans CJK SC
import mplfonts

_otf = Path(mplfonts.__file__).parent / "fonts" / "NotoSansCJKsc-Regular.otf"
fm.fontManager.addfont(str(_otf))
plt.rcParams["font.family"] = ["DejaVu Sans", "Noto Sans CJK SC"]
plt.rcParams["font.size"] = 8
plt.rcParams["axes.linewidth"] = 0.6

C_LANG = {"en": "#1f77b4", "fr": "#d62728", "zh": "#2ca02c"}
TAN = "#f5e9d9"
ORANGE = "#d95f02"
DARKRED = "#7f2704"
GREY_BOX = "#e8e4dc"
GREY_EDGE = "#6b6b6b"


# ---------------------------------------------------------------------------
# Figure 1: simplified attribution graphs (raw antonym en/fr/zh)
# ---------------------------------------------------------------------------
def load_sn(fmt):
    return {
        s["name"]: s
        for s in json.loads(
            (REPO / f"notebooks/supernodes/multilingual_{fmt}_4b.json").read_text()
        )["supernodes"]
    }


SN_RAW = load_sn("raw")
manifest = json.loads((SNI / "manifest.json").read_text())


def group_edges(gname: str, groups: dict[str, set[tuple[int, int]]]):
    """Aggregate signed pruned-graph edge weights between supernode groups (+ logit)."""
    gd = json.loads((SNI / f"graph_{gname}.json").read_text())
    nodes = gd["nodes"]
    owner = {}
    for gname2, feats in groups.items():
        for i, n in enumerate(nodes):
            if n["node_type"] == "feature" and (n["layer"], n["feature_idx"]) in feats:
                owner[i] = gname2
    ans_id = manifest["graphs"][gname]["answer_token_id"]
    for i, n in enumerate(nodes):
        if n["node_type"] == "logit" and n.get("token_id") == ans_id:
            owner[i] = "_logit"
    agg: dict[tuple[str, str], float] = defaultdict(float)
    for e in gd["edges"]:
        a, b = owner.get(e["source"]), owner.get(e["target"])
        if a and b and a != b:
            agg[(a, b)] += e["weight"]
    return agg


def draw_box(ax, x, y, lines, *, fc=GREY_BOX, ec="#555555", w=0.26, h=0.115, lw=1.0, fs=7.5):
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.012",
            fc=fc,
            ec=ec,
            lw=lw,
            zorder=3,
        )
    )
    if len(lines) == 1:
        ax.text(x, y, lines[0], ha="center", va="center", fontsize=fs, zorder=4)
    else:
        ax.text(x, y + 0.022, lines[0], ha="center", va="center", fontsize=fs, zorder=4)
        ax.text(
            x,
            y - 0.028,
            lines[1],
            ha="center",
            va="center",
            fontsize=fs - 1.3,
            color="#444444",
            zorder=4,
        )


def draw_edge(ax, p0, p1, w, wmax, *, positive=True):
    lw = 0.6 + 3.4 * min(abs(w) / wmax, 1.0)
    ax.add_patch(
        FancyArrowPatch(
            p0,
            p1,
            arrowstyle="-|>",
            mutation_scale=9,
            lw=lw,
            color=GREY_EDGE if positive else "#b2182b",
            linestyle="-" if positive else (0, (3, 2)),
            alpha=0.85,
            zorder=2,
            shrinkA=10,
            shrinkB=10,
        )
    )


def fig_graphs(fmt="raw"):
    sn_file = load_sn(fmt)
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.0))
    answers = {"en": "large", "fr": "grand", "zh": "大"}
    for ax, lg in zip(axes, ("en", "fr", "zh")):
        gname = f"antonym_{lg}" if fmt == "chat" else f"raw_antonym_{lg}"
        if fmt == "raw":
            layout = {
                f"opposite ({lg})": (0.18, 0.16),
                "small (multilingual)": (0.50, 0.16),
                f"quote ({lg})": (0.82, 0.16),
                "antonym (multilingual)": (0.18, 0.52),
                "say large (multilingual)": (0.50, 0.70),
                f"say large ({lg})": (0.82, 0.70),
                "_logit": (0.50, 0.94),
            }
        else:
            layout = {
                f"opposite ({lg})": (0.22, 0.16),
                "small (multilingual)": (0.62, 0.16),
                "antonym (multilingual)": (0.22, 0.52),
                "say large (multilingual)": (0.44, 0.70),
                f"say large ({lg})": (0.80, 0.70),
                "_logit": (0.50, 0.94),
            }
        groups: dict[str, set[tuple[int, int]]] = {}
        pos_map = {}
        for name in list(layout):
            if name == "_logit":
                continue
            sn = sn_file.get(name)
            members = (
                {
                    (m["layer"], m["feature"])
                    for m in sn["members"]
                    if gname in m.get("positions", {})
                }
                if sn
                else set()
            )
            if members:
                groups[name] = members
                pos_map[name] = layout[name]
        pos_map["_logit"] = layout["_logit"]
        agg = group_edges(gname, groups)
        wmax = max((abs(v) for v in agg.values()), default=1.0)
        for (a, b), v in sorted(agg.items(), key=lambda kv: -abs(kv[1])):
            if a == "_logit":
                continue
            thr = 0.005 if b == "_logit" else 0.04
            if abs(v) < thr * wmax:
                continue
            draw_edge(ax, pos_map[a], pos_map[b], v, wmax, positive=v >= 0)
        for name, (x, y) in pos_map.items():
            if name == "_logit":
                draw_box(ax, x, y, [f'"{answers[lg]}"'], fc="#ffffff", ec="#333333", w=0.2, h=0.09)
            else:
                base, _, suffix = name.partition(" (")
                sub = suffix.rstrip(")")
                n_mem = len(groups.get(name, []))
                draw_box(ax, x, y, [base, f"{sub} ({n_mem})"])
        prompt = manifest["graphs"][gname]["prompt"].replace('"', "″")
        if len(prompt) > 46:
            cut = prompt.rfind(" ", 0, 46)
            prompt = prompt[:cut] + "\n" + prompt[cut + 1 :]
        ax.set_title(f"{lg}: “{prompt}”", fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.04)
        ax.axis("off")
    fig.tight_layout()
    suffix = "" if fmt == "raw" else "_chat"
    fig.savefig(OUT / f"fig_ml_graphs{suffix}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"fig_ml_graphs{suffix}.pdf")


# ---------------------------------------------------------------------------
# Figure 2: interventions — box diagrams (measured readouts) + strength curves
# ---------------------------------------------------------------------------
op = json.loads((ART / "operation_results_constrained.json").read_text())
od = json.loads((ART / "operand_results_constrained.json").read_text())
lang = json.loads((ART / "language_results_constrained.json").read_text())
extra_op = json.loads((ART / "extra_operation_results.json").read_text())
op_ro = json.loads((ART / "operation_readouts_constrained.json").read_text())
od_ro = json.loads((ART / "operand_readouts_constrained.json").read_text())
lang_ro = json.loads((ART / "language_readouts_constrained.json").read_text())


def pct(ro_row, key):
    v = ro_row[key]
    return "pinned" if v["mean_pct"] is None else f"{v['mean_pct']:.0f}%"


def diagram(ax, spec):
    """One paper-style intervention diagram: boxes + badges + measured % labels."""
    for b in spec["boxes"]:
        x, y = b["xy"]
        fc = {"steer": TAN, "donor": TAN, "plain": GREY_BOX, "out": "#ffffff"}[
            b.get("kind", "plain")
        ]
        ec = ORANGE if b.get("kind") in ("steer", "donor") else "#555555"
        lw = 1.6 if b.get("kind") in ("steer", "donor") else 1.0
        draw_box(ax, x, y, b["lines"], fc=fc, ec=ec, w=b.get("w", 0.3), h=0.12, lw=lw)
        if "badge" in b:
            bx, by = x - b.get("w", 0.3) / 2 + 0.05, y + 0.105
            color = DARKRED if b["badge"].startswith("−") else ORANGE
            ax.add_patch(
                FancyBboxPatch(
                    (bx - 0.05, by - 0.026),
                    0.10,
                    0.052,
                    boxstyle="round,pad=0.006",
                    fc=color,
                    ec="none",
                    zorder=5,
                )
            )
            ax.text(
                bx, by, b["badge"], ha="center", va="center", fontsize=7, color="white", zorder=6
            )
        if "pct" in b:
            px = x + b.get("w", 0.3) / 2 - 0.01 if "badge" in b else x
            ax.text(
                px,
                y + 0.095,
                b["pct"],
                ha="right" if "badge" in b else "center",
                va="center",
                fontsize=6.5,
                color="#8a8a8a",
                zorder=6,
            )
    for a, b_, positive in spec["arrows"]:
        draw_edge(ax, a, b_, 1.0, 1.6, positive=positive)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.axis("off")
    ax.set_title(spec["title"], fontsize=8.5)


BEH = json.loads((ART / "behavior.json").read_text())
RBEH = json.loads((ART / "raw_behavior.json").read_text())
LANG_NAME = {"en": "English", "fr": "French", "zh": "Chinese"}
# Language-swap expected tokens are the paper's Fig. B5 outcomes, not the model's
# own raw answers (for en: "big", not "large") — must match the notebook's PAPER_OUT.
PAPER_OUT = {"zh": "大", "en": "big", "fr": "grand"}


def curve_panel(ax, r, base_tok, exp_tok, paper_s):
    """One notebook-style sweep panel: baseline vs expected with crossover marked."""
    ax.plot(
        r["strengths"],
        r["p_baseline"],
        "o-",
        ms=3,
        lw=1.2,
        color="#1f77b4",
        label=f"baseline {base_tok!r}",
    )
    ax.plot(
        r["strengths"],
        r["p_expected"],
        "s-",
        ms=3,
        lw=1.2,
        color="#ff7f0e",
        label=f"expected {exp_tok!r}",
    )
    ax.axvline(paper_s, color="grey", ls=":", lw=0.9)
    ax.text(paper_s, 0.97, " paper", color="grey", fontsize=6, va="top")
    if r["crossover"] is not None:
        ax.axvline(r["crossover"], color="red", ls="--", lw=0.9, alpha=0.75)
        ax.text(r["crossover"], 0.5, f" ×{r['crossover']:g}", color="red", fontsize=6.5)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25, lw=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=5.6, frameon=True, framealpha=0.85, borderpad=0.25)
    ax.tick_params(labelsize=6.5)


def out_label(r, paper_s, *, suffix=True):
    i = r["strengths"].index(paper_s)
    tok, prob = r["top_tokens"][i][0]
    if not suffix:
        return f"“{tok}” {float(prob):.2f}"
    note = f"flips at {r['crossover']:g}×" if r["crossover"] is not None else "no flip"
    return f"“{tok}” {float(prob):.2f} — {note}"


def _operation_spec(lg):
    ro = op_ro[f"chat_{lg}@6x"]
    return {
        "title": LANG_NAME[lg],
        "boxes": [
            {"xy": (0.15, 0.13), "lines": ["opposite", lg], "w": 0.24, "pct": "pinned"},
            {"xy": (0.46, 0.13), "lines": ["small", "multilingual"], "w": 0.26, "pct": "pinned"},
            {
                "xy": (0.24, 0.45),
                "lines": ["antonym", "multilingual"],
                "w": 0.28,
                "kind": "steer",
                "badge": "−5×",
                "pct": "pinned",
            },
            {
                "xy": (0.68, 0.45),
                "lines": ["synonym", "multilingual"],
                "w": 0.28,
                "kind": "donor",
                "badge": "+6×",
            },
            {
                "xy": (0.26, 0.75),
                "lines": ["say large", "multilingual"],
                "w": 0.27,
                "pct": pct(ro, "say_large_pct_baseline"),
            },
            {
                "xy": (0.70, 0.75),
                "lines": ["say small", "multilingual"],
                "w": 0.27,
                "pct": pct(ro, "say_small_pct_stored"),
            },
            {
                "xy": (0.48, 0.95),
                "lines": [out_label(op[f"chat_{lg}"], 6.0)],
                "kind": "out",
                "w": 0.52,
            },
        ],
        "arrows": [
            ((0.15, 0.19), (0.22, 0.39), True),
            ((0.46, 0.19), (0.27, 0.69), True),
            ((0.26, 0.51), (0.26, 0.69), True),
            ((0.68, 0.51), (0.70, 0.69), True),
            ((0.28, 0.81), (0.44, 0.90), True),
        ],
    }


def _operand_spec(lg):
    ro = od_ro[f"chat_{lg}@1.5x"]
    return {
        "title": LANG_NAME[lg],
        "boxes": [
            {"xy": (0.14, 0.13), "lines": ["opposite", lg], "w": 0.22, "pct": "pinned"},
            {
                "xy": (0.45, 0.13),
                "lines": ["small", "multilingual"],
                "w": 0.26,
                "kind": "steer",
                "badge": "−0.5×",
                "pct": "pinned",
            },
            {
                "xy": (0.80, 0.13),
                "lines": ["hot", "multilingual"],
                "w": 0.26,
                "kind": "donor",
                "badge": "+1.5×",
            },
            {
                "xy": (0.26, 0.75),
                "lines": ["say large", "multilingual"],
                "w": 0.27,
                "pct": pct(ro, "say_large_pct_baseline"),
            },
            {
                "xy": (0.70, 0.75),
                "lines": ["say cold", "multilingual"],
                "w": 0.27,
                "pct": pct(ro, "say_cold_pct_stored"),
            },
            {
                "xy": (0.48, 0.95),
                "lines": [out_label(od[f"chat_{lg}"], 1.5)],
                "kind": "out",
                "w": 0.52,
            },
        ],
        "arrows": [
            ((0.14, 0.19), (0.24, 0.69), True),
            ((0.45, 0.19), (0.28, 0.69), True),
            ((0.80, 0.19), (0.72, 0.69), True),
            ((0.28, 0.81), (0.44, 0.90), True),
        ],
    }


def _language_spec(a, b):
    key = f"{a}->{b}"
    ro = lang_ro[f"{key}@6x"]
    boxes = [
        {"xy": (0.13, 0.13), "lines": ["opposite", a], "w": 0.22, "pct": "pinned"},
        {
            "xy": (0.45, 0.13),
            "lines": ["quote", a],
            "w": 0.24,
            "kind": "steer",
            "badge": "−5×",
            "pct": "pinned",
        },
        {"xy": (0.80, 0.13), "lines": ["quote", b], "w": 0.24, "kind": "donor", "badge": "+6×"},
        {"xy": (0.16, 0.75), "lines": ["say large", "multiling."], "w": 0.24, "pct": "pinned"},
        {
            "xy": (0.5, 0.95),
            "lines": [out_label(lang[key], 6.0, suffix=False)],
            "kind": "out",
            "w": 0.5,
        },
    ]
    arrows = [((0.45, 0.19), (0.48, 0.69), False), ((0.28, 0.81), (0.42, 0.90), True)]
    if f"say_large_src_pct_baseline" in ro:
        boxes.append(
            {
                "xy": (0.48, 0.75),
                "lines": ["say large", a],
                "w": 0.24,
                "pct": pct(ro, "say_large_src_pct_baseline"),
            }
        )
        arrows.append(((0.48, 0.81), (0.48, 0.90), False))
    if f"say_large_tgt_pct_stored" in ro:
        boxes.append(
            {
                "xy": (0.83, 0.75),
                "lines": ["say large", b],
                "w": 0.24,
                "pct": pct(ro, "say_large_tgt_pct_stored"),
            }
        )
        arrows.append(((0.80, 0.19), (0.83, 0.69), True))
        arrows.append(((0.83, 0.81), (0.56, 0.90), True))
    return {"title": f"{a}→{b}", "boxes": boxes, "arrows": arrows}


def fig_swap_operation():
    fig, axes = plt.subplots(3, 3, figsize=(11.4, 8.6), height_ratios=[1.1, 1, 1])
    for ci, lg in enumerate(("en", "fr", "zh")):
        diagram(axes[0][ci], _operation_spec(lg))
        for ri, (fmt, beh) in enumerate((("chat", BEH), ("raw", RBEH)), start=1):
            r = op[f"{fmt}_{lg}"]
            curve_panel(
                axes[ri][ci],
                r,
                beh["antonym"][lg]["token"].strip(),
                beh["synonym"][lg]["token"].strip(),
                6.0,
            )
            if ci == 0:
                axes[ri][ci].set_ylabel(f"{fmt}\nnext-token probability", fontsize=7.5)
            if ri == 2:
                axes[ri][ci].set_xlabel("intervention strength (× donor act)", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ml_swap_operation.pdf", bbox_inches="tight")
    plt.close(fig)
    print("fig_ml_swap_operation.pdf")


def fig_swap_operand():
    fig, axes = plt.subplots(3, 3, figsize=(11.4, 8.6), height_ratios=[1.1, 1, 1])
    for ci, lg in enumerate(("en", "fr", "zh")):
        diagram(axes[0][ci], _operand_spec(lg))
        for ri, (fmt, beh) in enumerate((("chat", BEH), ("raw", RBEH)), start=1):
            r = od[f"{fmt}_{lg}"]
            curve_panel(
                axes[ri][ci],
                r,
                beh["antonym"][lg]["token"].strip(),
                beh["antonym_hot"][lg]["token"].strip(),
                1.5,
            )
            if ci == 0:
                axes[ri][ci].set_ylabel(f"{fmt}\nnext-token probability", fontsize=7.5)
            if ri == 2:
                axes[ri][ci].set_xlabel("intervention strength (× donor act)", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ml_swap_operand.pdf", bbox_inches="tight")
    plt.close(fig)
    print("fig_ml_swap_operand.pdf")


def fig_swap_language():
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 5.9), height_ratios=[1.1, 1])
    for ci, (a, b) in enumerate((("en", "zh"), ("fr", "en"), ("zh", "fr"))):
        diagram(axes[0][ci], _language_spec(a, b))
        r = lang[f"{a}->{b}"]
        curve_panel(
            axes[1][ci],
            r,
            RBEH["antonym"][a]["token"].strip(),
            PAPER_OUT[b],
            6.0,
        )
        axes[1][ci].set_xlabel("intervention strength (× donor act)", fontsize=7)
        if ci == 0:
            axes[1][ci].set_ylabel("raw\nnext-token probability", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ml_swap_language.pdf", bbox_inches="tight")
    plt.close(fig)
    print("fig_ml_swap_language.pdf")


def fig_qk_probe():
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 5.4))
    for ri, (variant, tag) in enumerate((("frozen", "patterns frozen"), ("live", "patterns live"))):
        for ci, lg in enumerate(("en", "fr", "zh")):
            r = extra_op["arm2_qk_probe"][f"chat_{lg}_{variant}"]
            curve_panel(
                axes[ri][ci],
                r,
                BEH["antonym"][lg]["token"].strip(),
                BEH["synonym"][lg]["token"].strip(),
                6.0,
            )
            if ri == 0:
                axes[ri][ci].set_title(LANG_NAME[lg], fontsize=8.5)
            if ci == 0:
                axes[ri][ci].set_ylabel(f"{tag}\nnext-token probability", fontsize=7.5)
            if ri == 1:
                axes[ri][ci].set_xlabel("intervention strength (× donor act)", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ml_qk_probe.pdf", bbox_inches="tight")
    plt.close(fig)
    print("fig_ml_qk_probe.pdf")


# ---------------------------------------------------------------------------
# Figure 3: overlap + scale
# ---------------------------------------------------------------------------
def fig_overlap():
    c4 = json.loads((ART / "overlap_curves.json").read_text())
    c8 = json.loads((REPO / "artifacts/paper_multilingual/8b/overlap_curves.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 2.55))
    colors = {"en-fr": "#1f77b4", "en-zh": "#2ca02c", "fr-zh": "#9467bd"}
    ax = axes[0]
    for pair, col in colors.items():
        ax.plot(c4[pair], color=col, lw=1.5, label=pair)
        ax.plot(
            c4[f"{pair}-baseline"], color=col, lw=1.1, ls="--", alpha=0.55, label=f"{pair} baseline"
        )
    ax.set_xlabel("layer")
    ax.set_ylabel("IOU of active-feature sets")
    ax.set_title("4B: translated pairs vs. unrelated-pair baseline", fontsize=8.5)
    ax.legend(fontsize=6, ncol=2, frameon=False)
    ax = axes[1]
    for pair, col in colors.items():
        ax.plot(
            np.array(c4[pair]) - np.array(c4[f"{pair}-baseline"]),
            color=col,
            lw=1.5,
            label=f"4B {pair}",
        )
        ax.plot(
            np.array(c8[pair]) - np.array(c8[f"{pair}-baseline"]),
            color=col,
            lw=1.5,
            ls="--",
            alpha=0.75,
            label=f"8B {pair}",
        )
    ax.set_xlabel("layer")
    ax.set_ylabel("IOU − baseline")
    ax.set_title("baseline-subtracted (the paper's quantity), 4B vs. 8B", fontsize=8.5)
    ax.legend(fontsize=6, ncol=2, frameon=False)
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_ml_overlap.pdf", bbox_inches="tight")
    plt.close(fig)
    print("fig_ml_overlap.pdf")


fig_graphs("raw")
fig_graphs("chat")
fig_swap_operation()
fig_swap_operand()
fig_swap_language()
fig_qk_probe()
fig_overlap()
print("done ->", OUT)
