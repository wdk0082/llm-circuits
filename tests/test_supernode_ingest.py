"""Unit tests for the addition-v3 supernode pipeline (build_supernodes.py).

Synthetic ``supernode_inputs`` root (mini manifest, four addition graph dumps + one
multilingual dump, crafted three-probe grids.npz) exercising the seed scan, the
review-page grid attachment, and the export ingest — CPU-only, model-free.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))

import addition_helper as A
import build_supernodes as BS

PAIR, DONOR = (46, 49), (49, 49)
UNION = [(2, 100), (3, 200), (4, 700), (5, 500), (10, 600), (20, 300), (21, 800), (30, 400)]
AA, BB = np.meshgrid(np.arange(100), np.arange(100), indexing="ij")


def _grids(probe: str) -> np.ndarray:
    by_feat = {
        "peak": {
            (2, 100): (AA % 10 == 6).astype(np.float32) * 3,
            (5, 500): (AA % 10 == 6).astype(np.float32) * 2,
            (3, 200): (BB % 10 == 9).astype(np.float32) * 4,
            (4, 700): ((AA >= 40) & (AA <= 52)).astype(np.float32),
        },
        "ones": {
            (20, 300): ((AA % 10 == 6) & (BB % 10 == 9)).astype(np.float32) * 5,
            (21, 800): ((AA % 10 == 9) & (BB % 10 == 9)).astype(np.float32) * 6,
            (30, 400): ((AA + BB) % 10 == 5).astype(np.float32),
        },
        "final": {
            (10, 600): (np.abs(AA + BB - 95) <= 4).astype(np.float32) * 2,
        },
    }[probe]
    return np.stack([by_feat.get(f, np.zeros((100, 100), np.float32)) for f in UNION])


def _feature(layer, feat, pos, act, influence=0.001):
    return {
        "node_type": "feature",
        "layer": layer,
        "feature_idx": feat,
        "position": pos,
        "activation": act,
        "influence": influence,
        "label": {"top_logits": [f"t{feat}"]},
    }


@pytest.fixture()
def world(tmp_path, monkeypatch):
    root = tmp_path / "supernode_inputs"
    root.mkdir()
    monkeypatch.setattr(BS, "SUPERNODE_DIR", tmp_path / "supernodes")

    graphs = {
        "ones": {
            "prompt": "calc: 46+49=9",
            "tokens": list("x" * 11),
            "nodes": [
                _feature(2, 100, 5, 6.0),
                _feature(5, 500, 5, 2.0),
                _feature(3, 200, 8, 5.0),
                _feature(20, 300, 10, 8.0),
                _feature(30, 400, 10, 4.0),
            ],
            "edges": [],
        },
        "first": {
            "prompt": "calc: 46+49=",
            "tokens": list("x" * 10),
            "nodes": [_feature(4, 700, 5, 3.0), _feature(10, 600, 9, 7.0)],
            "edges": [],
        },
        "donor": {
            "prompt": "calc: 49+49=9",
            "tokens": list("x" * 11),
            "nodes": [_feature(21, 800, 10, 9.0)],
            "edges": [],
        },
        "reuse": {
            "prompt": "founded in 1949 ... volume 46 appeared in 199",
            "tokens": list("x" * 13),
            "nodes": [_feature(20, 300, 12, 1.5)],
            "edges": [],
        },
        "antonym_en": {
            "prompt": "what is the opposite",
            "tokens": list("x" * 5),
            "nodes": [_feature(7, 900, 3, 2.0)],
            "edges": [],
        },
    }
    for name, gd in graphs.items():
        (root / f"graph_{name}.json").write_text(json.dumps(gd))

    dp = {"a_digits": [4, 5], "b_digits": [7, 8], "plus": 6, "eq": 9}
    manifest = {
        "size": "testsz",
        "git": "deadbee",
        "addition": {
            "pair": list(PAIR),
            "donor_pair": list(DONOR),
            "pair_note": "synthetic",
            "accuracy_pct": 50.0,
        },
        "graphs": {
            "ones": {
                "task": "addition",
                "target": "ones",
                "pair": list(PAIR),
                "final_position": 10,
                "digit_positions": dp,
            },
            "first": {
                "task": "addition",
                "target": "first",
                "pair": list(PAIR),
                "final_position": 9,
                "digit_positions": dp,
            },
            "donor": {
                "task": "addition",
                "target": "ones",
                "pair": list(DONOR),
                "final_position": 10,
                "digit_positions": dp,
            },
            "reuse": {
                "task": "addition",
                "target": "reuse-ones",
                "pair": list(PAIR),
                "final_position": 12,
            },
            "antonym_en": {"raw": False, "final_position": 4},
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest))

    np.savez_compressed(
        root / "grids.npz",
        a_vals=np.arange(100),
        b_vals=np.arange(100),
        **{f"{probe}_grids": _grids(probe) for probe in ("final", "ones", "peak")},
        **{
            f"{probe}_features": np.array(UNION, dtype=np.int64)
            for probe in ("final", "ones", "peak")
        },
    )
    ctx = BS.load_grid_ctx(root)
    return root, manifest, graphs, ctx


def _export(gname, groups):
    return {
        "example": gname,
        "groups": [
            {
                "name": name,
                "nodes": [
                    {"layer": layer, "feature_idx": feat, "position": pos}
                    for (layer, feat, pos) in nodes
                ],
            }
            for name, nodes in groups
        ],
    }


def _write_exports(tmp_path, exports):
    files = []
    for i, exp in enumerate(exports):
        p = tmp_path / f"groups_{exp['example']}_{i}.json"
        p.write_text(json.dumps(exp))
        files.append(str(p))
    return files


def _happy_exports(tmp_path):
    return _write_exports(
        tmp_path,
        [
            _export(
                "ones",
                [
                    ("input _6", [(2, 100, 5), (5, 500, 5)]),
                    ("input _9", [(3, 200, 8)]),
                    ("lookup (_6+_9)", [(20, 300, 10)]),
                    ("sum = _5", [(30, 400, 10)]),
                    ("(overflow) input _6", [(5, 500, 5)]),  # helper group -> skipped
                ],
            ),
            _export(
                "first",
                [
                    ("magnitude ~46", [(4, 700, 5)]),
                    ("sum ~95 (low precision)", [(10, 600, 9)]),
                ],
            ),
            _export("donor", [("lookup (_9+_9) donors", [(21, 800, 10)])]),
            _export("reuse", [("reuse lookups (_6+_9)", [(20, 300, 12)])]),
        ],
    )


class TestGridCtxAndNames:
    def test_ctx_reports_and_missing(self, world):
        _, _, _, ctx = world
        assert ctx.report("ones", 20, 300)[0] == "lookup(a%10=6,b%10=9)"
        assert ctx.report("peak", 2, 100)[0] == "mod10-a(r6)"
        assert ctx.report("final", 10, 600)[0].startswith("sum-band(~9")
        assert ctx.report("ones", 2, 100)[0] == "dead"  # zeros grid, present in union
        assert ctx.report("peak", 99, 1) is None  # not in the union
        assert ctx.grid("ones", 20, 300).shape == (100, 100)

    def test_canonical_names_and_specs(self, world):
        _, manifest, _, _ = world
        names = BS.addition_names(manifest)
        assert names["input _6"] == {
            "role": "source",
            "graph": "ones",
            "position": 5,
            "required": True,
            "note": "",
        }
        assert names["lookup (_9+_9) donors"]["graph"] == "donor"
        assert names["sum = _5"]["position"] == "final"
        assert names["reuse lookups (_6+_9)"]["role"] == "annotation"
        assert [n for n, s in names.items() if s["required"]] == [
            "input _6",
            "input _9",
            "sum ~95 (low precision)",
            "lookup (_6+_9)",
            "sum = _5",
            "lookup (_9+_9) donors",
        ]


class TestSeedBuilder:
    def test_seed_members_by_grid_class(self, world):
        root, manifest, graphs, ctx = world
        doc = BS.build_addition_seed("testsz", root, manifest, graphs, ctx)
        assert doc["approved"] is False and doc["task"] == "addition"
        by_name = {sn["name"]: sn for sn in doc["supernodes"]}
        assert {(m["layer"], m["feature"]) for m in by_name["input _6"]["members"]} == {
            (2, 100),
            (5, 500),
        }
        assert by_name["input _6"]["members"][0]["layer"] == 2  # earliest-first ranking
        assert [(m["layer"], m["feature"]) for m in by_name["lookup (_6+_9)"]["members"]] == [
            (20, 300)
        ]
        assert by_name["lookup (_6+_9)"]["members"][0]["evidence"]["grid_class"] == (
            "lookup(a%10=6,b%10=9)"
        )
        assert [(m["layer"], m["feature"]) for m in by_name["sum = _5"]["members"]] == [(30, 400)]
        assert [(m["layer"], m["feature"]) for m in by_name["magnitude ~46"]["members"]] == [
            (4, 700)
        ]
        assert by_name["lookup (_9+_9) donors"]["graph"] == "donor"
        assert [
            (m["layer"], m["feature"]) for m in by_name["reuse lookups (_6+_9)"]["members"]
        ] == [(20, 300)]
        assert by_name["input 46 (exact)"]["members"] == []  # no cross grid in the fixture


class TestAttachOperandGrids:
    def test_refs_store_and_non_mutation(self, world):
        _, manifest, graphs, ctx = world
        entry = manifest["graphs"]["ones"]
        out = BS.attach_operand_grids(graphs["ones"], "ones", entry, ctx)
        assert "operand_grid_store" not in graphs["ones"]
        assert all("operand_grids" not in (n.get("label") or {}) for n in graphs["ones"]["nodes"])

        lookup_node = next(n for n in out["nodes"] if n["feature_idx"] == 300)
        refs = lookup_node["label"]["operand_grids"]
        assert refs[0]["probe"] == "ones"  # final-position node on a ones-target graph
        assert refs[0]["cls"] == "lookup(a%10=6,b%10=9)"
        assert refs[0]["mark"] == [46, 49] and refs[0]["pair_frac"] == pytest.approx(1.0)
        input_node = next(n for n in out["nodes"] if n["feature_idx"] == 100)
        assert input_node["label"]["operand_grids"][0]["probe"] == "peak"  # operand position
        # every ref resolves in the store, and entries decode back to real grids
        from llm_circuits.circuits.grid_codec import decode_grid_u8

        for nd in out["nodes"]:
            for ref in (nd.get("label") or {}).get("operand_grids", []):
                assert ref["key"] in out["operand_grid_store"]
        g = decode_grid_u8(out["operand_grid_store"]["ones:L20f300"])
        assert g.shape == (100, 100) and g[46, 49] == pytest.approx(5.0, abs=0.02)


class TestIngest:
    def test_happy_path_writes_approved_file(self, world, tmp_path):
        root, manifest, graphs, ctx = world
        BS.ingest_addition_exports("testsz", _happy_exports(tmp_path), root, manifest, graphs, ctx)
        out = BS.SUPERNODE_DIR / "addition_testsz.json"
        doc = json.loads(out.read_text())
        assert doc["approved"] is True and doc["selection"].startswith("explorer-export")
        by_name = {sn["name"]: sn for sn in doc["supernodes"]}
        # canonical scalar graph/position; members keep their own int positions
        assert by_name["input _6"]["graph"] == "ones" and by_name["input _6"]["position"] == 5
        assert by_name["sum = _5"]["position"] == "final"
        m = by_name["lookup (_6+_9)"]["members"][0]
        assert m["position"] == 10 and m["review"] == "approved"
        assert m["source"] == "explorer-export"
        assert m["evidence"]["grid_class"] == "lookup(a%10=6,b%10=9)"
        assert m["evidence"]["on_pair_frac_of_max"] == pytest.approx(1.0)
        assert m["act"] == pytest.approx(8.0)  # act auto-filled from the dump node
        # donor evidence measured at the DONOR pair
        dm = by_name["lookup (_9+_9) donors"]["members"][0]
        assert dm["evidence"]["on_pair_frac_of_max"] == pytest.approx(1.0)
        # unexported annotations recorded as documented absences
        assert by_name["add ~49 (function)"]["members"] == []
        assert "absent after hand review" in by_name["add ~49 (function)"]["note"]
        # the notebook loader accepts the file (same feature on ones+reuse graphs is OK)
        sns = A.load_supernodes(out)
        assert {"input _6", "reuse lookups (_6+_9)"} <= set(sns)

    def test_custom_group_and_note(self, world, tmp_path):
        root, manifest, graphs, ctx = world
        exports = _happy_exports(tmp_path)
        extra = _export("ones", [("my hunch", [(5, 500, 5)])])
        # replace the ones export so 'my hunch' doesn't overlap input _6's claim
        ones = json.loads(Path(exports[0]).read_text())
        ones["groups"][0]["nodes"] = [{"layer": 2, "feature_idx": 100, "position": 5}]
        ones["groups"].append(extra["groups"][0])
        Path(exports[0]).write_text(json.dumps(ones))
        BS.ingest_addition_exports("testsz", exports, root, manifest, graphs, ctx)
        doc = json.loads((BS.SUPERNODE_DIR / "addition_testsz.json").read_text())
        custom = next(sn for sn in doc["supernodes"] if sn["name"] == "my hunch")
        assert custom["role"] == "custom" and custom["graph"] == "ones"
        assert custom["position"] == 5  # inferred from the single member position
        assert "wire before steering" in custom["note"]

    @pytest.mark.parametrize(
        ("mutate", "match"),
        [
            (
                lambda e: e[0]["groups"].__setitem__(0, dict(e[0]["groups"][0], name="renamed")),
                "required supernodes",
            ),
            (lambda e: e[1]["groups"].pop(0), "magnitude"),
            (
                lambda e: e[0]["groups"].append({"name": "sum ~95 (low precision)", "nodes": []}),
                "belongs on",
            ),
            (
                lambda e: e[0]["groups"].append(
                    {
                        "name": "also claims",
                        "nodes": [{"layer": 2, "feature_idx": 100, "position": 5}],
                    }
                ),
                "disjoint",
            ),
            (
                lambda e: e[3]["groups"].__setitem__(
                    0, dict(e[3]["groups"][0], name="lookup (_6+_9)")
                ),
                "belongs on",
            ),
        ],
    )
    def test_contract_violations_raise(self, world, tmp_path, mutate, match):
        root, manifest, graphs, ctx = world
        files = _happy_exports(tmp_path)
        exps = [json.loads(Path(f).read_text()) for f in files]
        mutate(exps)
        for f, exp in zip(files, exps, strict=True):
            Path(f).write_text(json.dumps(exp))
        with pytest.raises(SystemExit, match=match):
            BS.ingest_addition_exports("testsz", files, root, manifest, graphs, ctx)

    def test_multilingual_page_refused(self, world, tmp_path):
        root, manifest, graphs, ctx = world
        files = _write_exports(tmp_path, [_export("antonym_en", [("x", [(7, 900, 3)])])])
        with pytest.raises(SystemExit, match="not a dumped addition graph"):
            BS.ingest_addition_exports("testsz", files, root, manifest, graphs, ctx)

    def test_same_group_on_two_pages_refused(self, world, tmp_path):
        root, manifest, graphs, ctx = world
        files = _happy_exports(tmp_path)
        dup = _export("first", [("input _6", [(4, 700, 5)])])
        files += _write_exports(tmp_path, [dup])
        with pytest.raises(SystemExit, match="belongs on"):
            BS.ingest_addition_exports("testsz", files, root, manifest, graphs, ctx)


class TestEmitReviewHtmls:
    def test_grid_enabled_pages_with_overflow_groups(self, world, tmp_path):
        root, manifest, graphs, ctx = world
        doc = BS.build_addition_seed("testsz", root, manifest, graphs, ctx)
        doc["supernodes"][0]["overflow"] = [dict(doc["supernodes"][0]["members"][0])]
        BS.emit_review_htmls(root, manifest, graphs, [doc], ctx=ctx)
        page = (root / "review_ones.html").read_text()
        assert "(overflow) input _6" in page
        assert "ones:L20f300" in page  # grid store shipped
        assert "gridSectionHtml" in page  # grid panel JS present
        # addition pages emit even for graphs the seed left groupless
        assert (root / "review_reuse.html").exists()
        # multilingual page untouched by the grid path (no groups seeded here -> no page)
        assert not (root / "review_chat_antonym_en.html").exists()
