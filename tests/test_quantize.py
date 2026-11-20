"""Unit tests for src/quantize.py — round-trip error bounds per IEEE-754 dtype.

Expected relative error bounds (round-to-nearest-even):
  float32 : <= 2^-24 (eps/2), float16 : <= 2^-11, bfloat16 : <= 2^-8.
"""
import numpy as np
import pytest

from src import quantize


def test_float64_roundtrip_exact():
    x = np.random.default_rng(0).standard_normal((50, 40)) * 1e3
    out, scale = quantize.quantize_array(x, "float64")
    assert scale == 1.0 and np.array_equal(out, x)


def test_float32_roundtrip_bound():
    x = np.random.default_rng(1).standard_normal((200, 200)) * 1e5
    out, _ = quantize.quantize_array(x, "float32")
    rel = np.abs(out - x) / np.maximum(np.abs(x), 1e-300)
    assert rel.max() <= 2.0 ** -23  # one ulp bound, generous


def test_float16_values_are_exact_f16_and_bounded():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((300, 300)) * 1e6
    out, scale = quantize.quantize_array(x, "float16")
    assert scale == pytest.approx(65503.0 / np.abs(x).max())
    assert np.abs(out).max() <= 65504.0 and np.isfinite(out).all()
    # stored values must be exactly representable in float16
    assert np.array_equal(out, out.astype(np.float16).astype(np.float32))
    rel = np.abs(out / scale - x) / np.maximum(np.abs(x), 1e-300)
    assert rel.max() <= 2.0 ** -10  # f16 ulp is 2^-10 relative


def test_float16_no_rescale_when_small():
    x = np.array([[0.5, -1.25]])
    out, scale = quantize.quantize_array(x, "float16", rescale_max=65503.0)
    assert scale == 1.0
    assert np.array_equal(out, x.astype(np.float16).astype(np.float32))


def test_bfloat16_low_bits_zero_and_bounded():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((300, 300)) * 1e4
    out, scale = quantize.quantize_array(x, "bfloat16")
    assert scale == 1.0
    assert np.all(out.view(np.uint32) & np.uint32(0xFFFF) == 0)  # truncated to bf16
    rel = np.abs(out - x) / np.maximum(np.abs(x), 1e-300)
    assert rel.max() <= 2.0 ** -7  # bf16 ulp is 2^-7 relative


def test_bfloat16_round_to_nearest_even():
    # low-16-bit patterns of a float32 1.0 (0x3F800000):
    #  0x0001 -> down, 0x7FFF -> down, 0x8000 -> tie to even (down), 0x8001 -> up
    one = np.float32(1.0)
    u = int(one.view(np.uint32))
    vals = [(u + d) .view(np.float32) if False else np.array([0], np.uint32)[0] for d in []]
    xs = np.array([np.uint32(u + d).view(np.float32) for d in (0x0001, 0x7FFF, 0x8000, 0x8001)],
                  dtype=np.float32)
    out, _ = quantize.quantize_array(xs, "bfloat16")
    got = [int(o.view(np.uint32)) for o in out]
    assert got == [0x3F800000, 0x3F800000, 0x3F800000, 0x3F810000]


def test_visibilities_complex():
    rng = np.random.default_rng(4)
    v = (rng.standard_normal((100, 8)) + 1j * rng.standard_normal((100, 8))) * 50
    out, scale = quantize.quantize_visibilities(v, "float16")
    assert out.dtype == np.complex64
    assert np.abs(out).max() <= 65503.0
    re = out.real.astype(np.float16).astype(np.float32)
    assert np.array_equal(out.real, re)


def test_weights_rescale_by_sum():
    w = np.full((10, 4), 1e8)
    out, scale = quantize.quantize_weights(w, "float16")
    assert scale == pytest.approx(65503.0 / (40 * 1e8))
    assert np.abs(out).max() <= 65503.0


def test_invalid_dtype_raises():
    with pytest.raises(ValueError):
        quantize.quantize_array(np.zeros(3), "int8")


def test_f16_global_scale_not_block_floating_point():
    # anti-BFP: ONE global scalar scale per array (65503/amax). A block-floating-
    # point scheme would use a per-block scale and give small values full f16
    # relative precision; here they must not.
    x = np.concatenate([[1e6], np.geomspace(1e-5, 1e-3, 200)])
    out, scale = quantize.quantize_array(x, "float16")
    assert np.isscalar(scale) and scale == pytest.approx(65503.0 / x.max())
    # outputs exactly representable in f16 (a no-op f32 cast would fail this)
    assert np.array_equal(out, out.astype(np.float16).astype(np.float32))
    # error at max is f16-level (half-ulp ~2^-11 rel), not f32-level (2^-24)
    rel0 = abs(out[0] / scale - x[0]) / x[0]
    assert 1e-6 < rel0 <= 2.0 ** -10
    rel = np.abs(out[1:] / scale - x[1:]) / x[1:]
    # anti-BFP: x=1e-4 scaled by the GLOBAL max (0.0655) lands at 6.6e-6, below
    # the f16 normal floor (6.1e-5) -> subnormal, ulp 2^-24 scaled -> rel error
    # ~1e-2, far above f16 eps 2^-11. A per-block (BFP) scheme would give it
    # ~2^-11 relative precision instead.
    # the subnormal-region samples (x*scale < 2^-14) must show errors far above
    # f16 eps at least once -> the scale is global, not per-block
    assert rel.max() > 10 * 2.0 ** -11
