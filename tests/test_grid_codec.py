"""Round-trip tests for the uint8 grid codec used by the explorer's grid-plot panel."""

from __future__ import annotations

import base64
import zlib

import numpy as np
import pytest

from llm_circuits.circuits.grid_codec import decode_grid_u8, encode_grid_u8


def _lattice_grid() -> np.ndarray:
    """A mod-10-style operand grid: sparse, structured — the compressible case."""
    g = np.zeros((100, 100), dtype=np.float32)
    g[6::10, 9::10] = 5.0  # lookup-table lattice (a%10=6, b%10=9)
    g[50, :] = 1.25  # one operand band
    return g


class TestEncodeGridU8:
    def test_round_trip_error_bounded(self):
        rng = np.random.default_rng(0)
        g = rng.random((40, 60), dtype=np.float32) * 7.3
        for compress in (True, False):
            d = encode_grid_u8(g, compress=compress)
            assert d["enc"] == ("u8z" if compress else "u8")
            assert d["shape"] == [40, 60]
            assert d["vmax"] == pytest.approx(float(g.max()))
            back = decode_grid_u8(d)
            assert back.shape == (40, 60) and back.dtype == np.float32
            # uint8 quantization: worst case is half a step of vmax/255
            assert np.max(np.abs(back - g)) <= d["vmax"] / 255 / 2 + 1e-6

    def test_compression_shrinks_structured_grid(self):
        g = _lattice_grid()
        dz, d = encode_grid_u8(g, compress=True), encode_grid_u8(g, compress=False)
        assert len(dz["data"]) < len(d["data"]) / 5  # sparse lattice compresses hard
        # and the payload is genuine zlib-wrapped deflate (what DecompressionStream expects)
        raw = zlib.decompress(base64.b64decode(dz["data"]))
        assert raw == base64.b64decode(d["data"])
        np.testing.assert_allclose(decode_grid_u8(dz), decode_grid_u8(d))

    def test_zero_grid_encodes_empty(self):
        d = encode_grid_u8(np.zeros((4, 5)))
        assert d == {"enc": "u8", "shape": [4, 5], "vmax": 0.0, "data": ""}
        assert decode_grid_u8(d).shape == (4, 5)
        assert not decode_grid_u8(d).any()

    def test_nonfinite_and_negative_treated_inactive(self):
        g = np.array([[np.nan, -3.0], [np.inf, 2.0]])
        d = encode_grid_u8(g)
        back = decode_grid_u8(d)
        assert d["vmax"] == pytest.approx(2.0)
        assert back[0, 0] == 0.0 and back[0, 1] == 0.0 and back[1, 0] == 0.0
        assert back[1, 1] == pytest.approx(2.0, abs=2.0 / 255)

    def test_non_2d_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            encode_grid_u8(np.zeros(5))
        with pytest.raises(ValueError, match="unknown grid encoding"):
            decode_grid_u8({"enc": "f16", "shape": [1, 1], "vmax": 1.0, "data": "AA=="})

    def test_accepts_plain_lists(self):
        d = encode_grid_u8([[0.0, 1.0], [2.0, 4.0]])
        np.testing.assert_allclose(decode_grid_u8(d), [[0, 1], [2, 4]], atol=4 / 255)
