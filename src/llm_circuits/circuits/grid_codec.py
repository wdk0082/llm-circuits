"""Compact codec for 2-D activation heatmaps ("grid plots") embedded in graph labels.

The graph explorer can render a per-feature heatmap panel (e.g. the paper's operand
plots for addition: a feature's activation over every ``a+b=`` prompt, a,b in 0..99).
Shipping ~10k float values per feature in the page JSON is prohibitive at
1,000+ feature nodes, so grids are quantized to uint8 against their own maximum,
optionally zlib-compressed, and base64-encoded:

``{"enc": "u8z"|"u8", "shape": [n_rows, n_cols], "vmax": float, "data": base64 str}``

The explorer JS decodes ``u8z`` with the built-in ``DecompressionStream("deflate")``
(Python's ``zlib.compress`` emits zlib-wrapped deflate, which is exactly that format)
and dequantizes as ``q / 255 * vmax``. Quantization error is at most ``vmax / 510``.

This module is task-agnostic: what the rows/columns mean, which grids to attach, and
any derived statistics are the caller's business (see the supernode-review pipeline in
``notebooks/``). Non-finite values are treated as inactive (mapped to 0), and negative
values are clamped to 0 — transcoder activations are non-negative by construction.
"""

from __future__ import annotations

import base64
import zlib

import numpy as np


def encode_grid_u8(grid, *, compress: bool = True) -> dict:
    """Encode a 2-D array into the compact ``u8``/``u8z`` grid dict (see module docs).

    ``compress=True`` (default) zlib-compresses the quantized bytes (``enc: "u8z"``);
    structured activation grids typically shrink 5-20x. An all-zero (or empty) grid
    encodes as ``vmax: 0.0`` with empty ``data``.
    """
    a = np.asarray(grid, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(f"expected a 2-D grid, got shape {a.shape}")
    shape = [int(a.shape[0]), int(a.shape[1])]
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    a = np.clip(a, 0.0, None)
    vmax = float(a.max()) if a.size else 0.0
    if vmax <= 0.0:
        return {"enc": "u8", "shape": shape, "vmax": 0.0, "data": ""}
    q = np.round(a * (255.0 / vmax)).astype(np.uint8)
    raw = q.tobytes()  # row-major
    if compress:
        return {
            "enc": "u8z",
            "shape": shape,
            "vmax": vmax,
            "data": base64.b64encode(zlib.compress(raw, 6)).decode("ascii"),
        }
    return {
        "enc": "u8",
        "shape": shape,
        "vmax": vmax,
        "data": base64.b64encode(raw).decode("ascii"),
    }


def decode_grid_u8(d: dict) -> np.ndarray:
    """Decode an ``encode_grid_u8`` dict back to a float32 array (dequantized)."""
    shape = tuple(int(s) for s in d["shape"])
    if not d.get("data"):
        return np.zeros(shape, dtype=np.float32)
    raw = base64.b64decode(d["data"])
    enc = d.get("enc")
    if enc == "u8z":
        raw = zlib.decompress(raw)
    elif enc != "u8":
        raise ValueError(f"unknown grid encoding {enc!r}")
    q = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
    return q.astype(np.float32) * (float(d["vmax"]) / 255.0)
