#!/usr/bin/env python3
"""Convert a Qwen3.8/3.5 hybrid bf16 GGUF to ROCmFP4 (STRIX_LEAN recipe).

Recipe (reverse-engineered from Qwen3.8-27B-Q4_0_ROCMFP4_STRIX_LEAN.gguf):
  - token_embd.weight          -> Q6_K
  - attn_qkv (GDN fused) / attn_k / attn_v -> Q4_0_ROCMFP4     (2-scale layout)
  - all other quantizable weights -> Q4_0_ROCMFP4_FAST         (incl. ssm_alpha/beta)
  - norms / biases / ssm_a / ssm_dt / ssm_conv1d -> F32

Differences vs convert-jormungandr-rocmfp4.py:
  1. Fused attn_qkv is non-FAST (old script only matched separate k/v, so a
     fused qkv tensor got FASTed — Q component included).
  2. Quantization gate is TOTAL element count % 32 (not row dim), so tensors
     like ssm_alpha [5120, 48] get FASTed instead of staying F32.
  3. F32 whitelist is exact-ish (ssm_a / ssm_dt / ssm_conv1d), not the
     'ssm_a' substring that also caught ssm_alpha.
  4. general.file_type = 106 (vs 105).

Usage:
    python convert-strix-lean-rocmfp4.py <bf16-gguf> <out-gguf> [nthreads] [arch]
"""

import sys
import os
import ctypes
import numpy as np
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'gguf-py'))
from gguf import GGUFReader, GGUFWriter, GGUFValueType
from gguf.constants import GGMLQuantizationType

# ---------------------------------------------------------------------------
# libggml-base loading (for quantization C references)
# ---------------------------------------------------------------------------
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
            raise RuntimeError('libggml-base.so not found (run make first)')
    return _LIB

def quantize_rocmfp4_flat(data_f32):
    """Quantize a flat float32 array to ROCmFP4 (18 bytes per 32 values)."""
    lib = _lib()
    fn = lib.rocmfp4_quantize_row_q4_0_ref
    fn.restype = None
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong]
    n = data_f32.size
    assert n % 32 == 0, f'{n} not a multiple of 32'
    out_size = n * 18 // 32
    out = np.empty(out_size, dtype=np.uint8)
    fn(data_f32.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p), n)
    return out

def quantize_rocmfp4_fast_flat(data_f32):
    """Quantize a flat float32 array to ROCmFP4_FAST (17 bytes per 32 values)."""
    lib = _lib()
    fn = lib['rocmfp4_quantize_row_q4_0_fast_ref']
    fn.restype = None
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong]
    n = data_f32.size
    assert n % 32 == 0, f'{n} not a multiple of 32'
    out_size = n * 17 // 32
    out = np.empty(out_size, dtype=np.uint8)
    fn(data_f32.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p), n)
    return out

def quantize_q6k_flat(data_f32):
    """Quantize a flat float32 array to Q6_K."""
    lib = _lib()
    fn = lib.quantize_row_q6_K_ref
    fn.restype = None
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong]
    n = data_f32.size
    assert n % 256 == 0, f'{n} not a multiple of 256'
    out_size = n // 256 * 210
    out = np.empty(out_size, dtype=np.uint8)
    fn(data_f32.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p), n)
    return out

def read_source_data(src_ti):
    """Read source tensor data as a flat float32 array, one value per logical element.

    File dims are logical; for BF16 the reader hands us the raw uint8 byte
    stream (2 bytes per element). Convert the bf16 bit patterns to f32 via
    the classic `u16 << 16` trick (bf16 is the high half of f32; NOT
    interchangeable with f16 bit layouts).
    """
    raw = np.array(src_ti.data)  # ensure regular array (not memmap)
    t = int(src_ti.tensor_type)
    if t in (GGMLQuantizationType.BF16, 30):
        u16 = raw.reshape(-1)[: src_ti.n_elements * 2].view(np.uint16)
        return (u16.astype(np.uint32) << 16).view(np.float32).reshape(-1)
    return raw.astype(np.float32).reshape(-1)

# ---------------------------------------------------------------------------
# Main conversion logic
# ---------------------------------------------------------------------------
if len(sys.argv) < 3:
    sys.exit(f'''Usage: {Path(sys.argv[0]).name} <bf16-gguf> <out-gguf> [nthreads] [arch]

  <bf16-gguf>  source bf16 GGUF (Qwen3.8/3.5 hybrid)
  <out-gguf>   output ROCmFP4 GGUF
  [nthreads]   quantize threads (default 4)
  [arch]       output arch tag (default qwen35)

Example:
  {Path(sys.argv[0]).name} Qwen3.8-27B-bf16.gguf Qwen3.8-27B-Q4_0_ROCMFP4_STRIX_LEAN.gguf 8''')

BF16_GGUF = sys.argv[1]
OUT_GGUF = sys.argv[2]
N_THREADS = int(sys.argv[3]) if len(sys.argv) > 3 else 4
ARCH = sys.argv[4] if len(sys.argv) > 4 else 'qwen35'

r = GGUFReader(BF16_GGUF, 'r')
print(f'Reading {BF16_GGUF} ({len(r.tensors)} tensors)')

QK_ROCMFP4 = 32

def is_f32_only(name, shape):
    """STRIX_LEAN F32 whitelist: norms, biases, GDN scalars, conv1d, 1-D."""
    if len(shape) == 1:
        return True
    if '_norm.weight' in name or name.endswith('.norm.weight'):
        return True
    if 'ssm_conv1d' in name:
        return True
    if name.endswith('ssm_a') or 'ssm_dt' in name:
        return True
    if name.endswith('.bias'):
        return True
    return False

def quant_type_for(name, shape):
    if name == 'token_embd.weight':
        return 'Q6_K'
    if (name.endswith('.attn_qkv.weight')
            or name.endswith('.attn_k.weight')
            or name.endswith('.attn_v.weight')):
        return 'Q4_0_ROCMFP4'
    return 'Q4_0_ROCMFP4_FAST'

# Categorize tensors
tensors_info = []  # list of (name, logical_shape, should_quantize, quant_type)
for ti in r.tensors:
    name = ti.name
    shape = [int(x) for x in ti.shape]

    if is_f32_only(name, shape):
        tensors_info.append((name, shape, False, 'F32'))
        continue
    if ti.n_elements % QK_ROCMFP4 != 0:
        tensors_info.append((name, shape, False, 'F32'))
        continue

    tensors_info.append((name, shape, True, quant_type_for(name, shape)))

type_counts = Counter(t[3] for t in tensors_info)
print(f'Tensor type counts:')
for k, v in sorted(type_counts.items(), key=lambda x: -x[1]):
    print(f'  {k}: {v}')

# Write output GGUF
w = GGUFWriter(OUT_GGUF, ARCH)

# Copy metadata from source
for key, f in r.fields.items():
    if key in ('GGUF.version', 'GGUF.tensor_count', 'GGUF.kv_count'):
        continue
    t = f.types
    vtype = t[0]
    if key == 'general.file_type':
        w.add_key_value(key, 106, vtype)
        continue
    if vtype == GGUFValueType.ARRAY:
        w.add_key_value(key, f.contents(), vtype, sub_type=t[1])
    else:
        val = f.contents()
        w.add_key_value(key, val.tolist() if isinstance(val, np.ndarray) else val, vtype)

print('Metadata copied')

# Build per-tensor jobs: quantize in a thread pool (ctypes releases the GIL),
# write results back in the original order to keep the file layout stable.
src_by_name = {ti.name: ti for ti in r.tensors}

def quantize_job(job):
    name, shape, _should, qtype = job
    data = read_source_data(src_by_name[name])
    ne0 = int(shape[0])
    ne1 = int(shape[-1])
    if qtype == 'Q6_K':
        quantized = quantize_q6k_flat(data)
        tsz, blk = 210, 256
        raw_dtype = GGMLQuantizationType.Q6_K
    elif qtype == 'Q4_0_ROCMFP4':
        quantized = quantize_rocmfp4_flat(data)
        tsz, blk = 18, 32
        raw_dtype = GGMLQuantizationType.Q4_0_ROCMFP4
    else:
        quantized = quantize_rocmfp4_fast_flat(data)
        tsz, blk = 17, 32
        raw_dtype = GGMLQuantizationType.Q4_0_ROCMFP4_FAST
    # numpy-order byte shape (the writer reverses it back to gguf order and
    # converts to the quant shape, so the stored dims come out as [ne0, ne1])
    assert (ne0 * tsz) % blk == 0
    return name, quantized, [ne1, ne0 * tsz // blk], raw_dtype

quant_jobs = [t for t in tensors_info if t[2]]
results = {}
with ThreadPoolExecutor(max_workers=max(1, N_THREADS)) as pool:
    for idx, res in enumerate(pool.map(quantize_job, quant_jobs), 1):
        name, quantized, byte_shape, raw_dtype = res
        results[name] = (quantized, byte_shape, raw_dtype)
        if idx % 50 == 0:
            print(f'  quantized {idx}/{len(quant_jobs)} tensors')

written = 0
for name, shape, should_quantize, qtype in tensors_info:
    if not should_quantize or qtype == 'F32':
        # Read as float32 and write directly (numpy-order shape so the
        # writer stores the same gguf-order dims as the source)
        data = read_source_data(src_by_name[name])
        w.add_tensor(name, data, raw_shape=list(reversed(shape)), raw_dtype=GGMLQuantizationType.F32)
    else:
        quantized, byte_shape, raw_dtype = results[name]
        w.add_tensor(name, quantized, raw_shape=byte_shape, raw_dtype=raw_dtype)

    written += 1
    if written % 50 == 0:
        print(f'  {written}/{len(tensors_info)} tensors written')

assert written == len(r.tensors)

# Actually write the GGUF file to disk (add_tensor only stores data in memory)
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f'\nDone: {OUT_GGUF} ({os.path.getsize(OUT_GGUF)/1e9:.2f} GB, {written} tensors)')
