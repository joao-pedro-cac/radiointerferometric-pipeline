"""Vectorized IEEE-754 quantization of pipeline arrays.

Pure functions: arrays in, (array, scale) out. No I/O.

Dtypes: "float64" | "float32" | "float16" | "bfloat16".
float16 has a narrow range (max 65504), so by default arrays are rescaled to
`rescale_max` before casting and the scale is returned for de-scaling
(legacy behaviour). Set rescale_max=None to cast raw (overflow -> inf).
bfloat16 = float32 with the low 16 mantissa bits rounded-to-nearest-even.
"""
import numpy as np

DTYPES = ("float64", "float32", "float16", "bfloat16")


def _check(dtype):
    if dtype not in DTYPES:
        raise ValueError(f"unknown dtype {dtype!r}, expected one of {DTYPES}")


def quantize_array(arr, dtype, rescale_max=65503.0):
    """Cast a real or complex array to `dtype`. Returns (cast_array, scale)."""
    _check(dtype)
    arr = np.asarray(arr)
    if dtype == "float64":
        return arr.astype(np.float64), 1.0
    if dtype == "float32":
        return arr.astype(np.float32), 1.0

    scale = 1.0
    a = arr.astype(np.float32)
    if dtype == "float16":
        amax = float(np.max(np.abs(a))) if a.size else 0.0
        if rescale_max is not None and amax > rescale_max:
            scale = rescale_max / amax
            a = a * np.float32(scale)
        out = a.astype(np.float16).astype(np.float32)   # store as f32, values are exact f16
        return out, scale
    if dtype == "bfloat16":
        # round-to-nearest-even truncation of the low 16 bits of float32
        u = a.view(np.uint32)
        rounded = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)
        out = (rounded << np.uint32(16)).view(np.float32)
        if np.iscomplexobj(arr):
            re = _bf16(a.real)
            im = _bf16(a.imag)
            return (re + 1j * im).astype(np.complex64), scale
        return out, scale
    raise AssertionError("unreachable")


def _bf16(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rounded = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) >> np.uint32(16)
    return (rounded << np.uint32(16)).view(np.float32)


def quantize_visibilities(vis, dtype, rescale_max=65503.0, scale=None):
    """Quantize complex visibilities. If `scale` is given (re-quantization of
    already-scaled visibilities), reuse it instead of re-estimating."""
    _check(dtype)
    vis = np.asarray(vis)
    if dtype in ("float64", "float32"):
        return vis.astype(np.complex128 if dtype == "float64" else np.complex64), 1.0
    if scale is None:
        amax = float(np.max(np.abs(vis))) if vis.size else 0.0
        scale = (rescale_max / amax) if (rescale_max is not None and amax > rescale_max) else 1.0
    scaled = (vis * np.complex64(scale)).astype(np.complex64)
    if dtype == "float16":
        out = scaled.astype(np.complex64)
        out.real = out.real.astype(np.float16).astype(np.float32)
        out.imag = out.imag.astype(np.float16).astype(np.float32)
        return out, scale
    # bfloat16
    re = _bf16(scaled.real)
    im = _bf16(scaled.imag)
    return (re + 1j * im).astype(np.complex64), scale


def quantize_image(img, dtype, rescale_max=65503.0):
    return quantize_array(img, dtype, rescale_max)


def quantize_weights(wgt, dtype, rescale_max=65503.0):
    _check(dtype)
    wgt = np.asarray(wgt, dtype=np.float64)
    if dtype == "float64":
        return wgt, 1.0
    if dtype == "float32":
        return wgt.astype(np.float32), 1.0
    scale = 1.0
    if dtype == "float16":
        s = float(np.sum(wgt))
        if rescale_max is not None and s > rescale_max:
            scale = rescale_max / s
        out = (wgt * scale).astype(np.float32)
        out = out.astype(np.float16).astype(np.float32)
        return out, scale
    out = _bf16(wgt)
    return out, scale
