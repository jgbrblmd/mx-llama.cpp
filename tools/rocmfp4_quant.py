#!/usr/bin/env python3
"""ROCmFP4 quantization via the fork's C reference (bit-identical).

Wraps rocmfp4_quantize_row_q4_0_ref from libggml-base.so so converted GGUFs
match llama-quantize output byte for byte. The C reference picks the UE4M3
scale per 16-value half block by exhaustive MSE search (Codebook10 values
{0,1,2,3,4,6,8,10}), then packs 2 nibbles per byte.
"""
import ctypes
import numpy as np
import torch
from pathlib import Path

_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        here = Path(__file__).resolve().parent
        for p in [here / '..' / 'build-hip' / 'bin' / 'libggml-base.so',
                  here / '..' / 'build' / 'bin' / 'libggml-base.so']:
            if p.exists():
                _LIB = ctypes.CDLL(str(p))
                break
        if _LIB is None:
            raise RuntimeError('libggml-base.so not found (build first)')
    return _LIB


def quantize_rocmfp4(t):
    """float32 tensor [..., n] -> uint8 ROCmFP4 bytes (18 bytes per 32 values).

    Bit-identical to quantize_row_rocmfp4_q4_0_ref (ggml/rocmfp4/rocmfp4.c).
    """
    fn = _lib().rocmfp4_quantize_row_q4_0_ref
    fn.restype = None
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong]
    xf = t.contiguous().reshape(-1).numpy().astype(np.float32)
    n = xf.size
    assert n % 32 == 0, f'{n} not a multiple of 32'
    out = np.empty(n * 18 // 32, dtype=np.uint8)
    fn(xf.ctypes.data, out.ctypes.data, n)
    return torch.from_numpy(out)
