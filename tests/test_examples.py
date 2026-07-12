"""Guard the ``examples/`` scripts against package API drift.

The scripts need a GPU, the 4b weights, and ~57 GB of transcoders, so CI cannot run
them. What CI *can* do is check that every call they make into ``llm_circuits`` still
type-checks against the real signatures: this catches a renamed parameter or a moved
function — the way a demo silently rots — without importing a single weight.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path

import pytest

EXAMPLES = sorted((Path(__file__).resolve().parents[1] / "examples").glob("*.py"))


def _imported_from_package(tree: ast.Module) -> dict[str, tuple[str, str]]:
    """Map local name -> (module, attribute) for every ``llm_circuits`` import."""
    out: dict[str, tuple[str, str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("llm_circuits"):
            for alias in node.names:
                out[alias.asname or alias.name] = (node.module or "", alias.name)
    return out


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.name)
def test_example_imports_resolve(script: Path) -> None:
    tree = ast.parse(script.read_text())
    for local, (module, attr) in _imported_from_package(tree).items():
        mod = importlib.import_module(module)
        assert hasattr(mod, attr), f"{script.name}: {module}.{attr} no longer exists ({local})"


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda p: p.name)
def test_example_calls_match_signatures(script: Path) -> None:
    tree = ast.parse(script.read_text())
    imports = _imported_from_package(tree)
    checked = 0

    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        target = imports.get(node.func.id)
        if target is None:
            continue
        fn = getattr(importlib.import_module(target[0]), target[1])
        if not callable(fn):
            continue
        params = inspect.signature(fn).parameters
        where = f"{script.name}:{node.lineno} {node.func.id}()"

        var_kw = any(p.kind is p.VAR_KEYWORD for p in params.values())
        for kw in node.keywords:
            if kw.arg is None:  # **kwargs unpacking — nothing static to check
                continue
            assert var_kw or kw.arg in params, f"{where}: no parameter {kw.arg!r}"

        if not any(isinstance(a, ast.Starred) for a in node.args):
            var_pos = any(p.kind is p.VAR_POSITIONAL for p in params.values())
            slots = sum(
                p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params.values()
            )
            assert var_pos or len(node.args) <= slots, (
                f"{where}: {len(node.args)} positional args, but only {slots} accepted"
            )
        checked += 1

    assert checked, f"{script.name}: no llm_circuits calls found — did the imports move?"


def test_demo_loads_decoders_eagerly_by_default() -> None:
    """The demo runs ~30 interventions; a lazy decoder makes each ~50x slower.

    Regression guard for the CLAUDE.md perf note: ``--lazy-decoder`` must stay opt-in.
    """
    demo = (Path(__file__).resolve().parents[1] / "examples" / "demo.py").read_text()
    tree = ast.parse(demo)

    lazy_flag = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "add_argument"
        and any(isinstance(a, ast.Constant) and a.value == "--lazy-decoder" for a in n.args)
    ]
    assert lazy_flag, "demo.py should expose --lazy-decoder as the small-VRAM escape hatch"
    assert any(
        kw.arg == "action" and kw.value.value == "store_true"  # type: ignore[attr-defined]
        for kw in lazy_flag[0].keywords
    ), "--lazy-decoder must default to off (store_true), i.e. eager decoders by default"

    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "load_transcoder"
    ]
    assert calls, "demo.py no longer calls load_transcoder"
    assert any(kw.arg == "lazy_decoder" for kw in calls[0].keywords), (
        "demo.py must pass lazy_decoder explicitly (the loader default is lazy=True)"
    )
