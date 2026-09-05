#!/usr/bin/env python3
"""Convert a Qwen35 MoE/dense bf16 GGUF to ROCmFP4 or Q8_0_ROCMFPX.

Recipes (reverse-engineered from llama-quant.cpp routing logic and verified
against KAT-Coder-V2.5-Dev-Q4_0_ROCMFP4.gguf):

ROCmFP4 recipe (--output rocmfp4, file_type=100):
  default type: Q4_0_ROCMFP4
  token_embd / output              -> Q6_K
  attn_qkv                         -> Q5_K
  attn_v                           -> Q5_K
  ffn_down_exps/shexp              -> Q6_K (blk < 2/3*n_layer) / Q5_K (else)
  ffn_gate_exps/shexp              -> Q5_K on "use_more_bits" layers, else Q4_0_ROCMFP4
  everything else quantizable      -> Q4_0_ROCMFP4 (default)

ROCmFP4 Strix Lean recipe (--output strix_leon, file_type=106):
  default type: Q4_0_ROCMFP4_FAST
  token_embd / output              -> Q5_K
  attn_qkv                         -> Q4_0_ROCMFP4
  attn_v                           -> Q4_0_ROCMFP4
  attn_k                           -> Q4_0_ROCMFP4
  ffn_down_exps/shexp              -> Q4_0_ROCMFP4_FAST
  ffn_up_exps/shexp                -> Q4_0_ROCMFP4_FAST
  ffn_gate_exps/shexp              -> Q4_0_ROCMFP4_FAST
  everything else quantizable      -> Q4_0_ROCMFP4_FAST

Q8_0_ROCMFPX recipe (--output rocmfpx, file_type=111):
  default type: Q8_0_ROCMFPX
  token_embd / output              -> Q8_0_ROCMFPX
  attn_qkv                         -> Q8_0
  attn_v / attn_k / attn_q         -> Q8_0_ROCMFPX
  attn_output                      -> Q8_0_ROCMFPX
  ffn_down_exps/shexp              -> Q8_0 (blk < n_layer/8 or blk >= 3*n_layer/4) else Q8_0_ROCMFPX
  ffn_up_exps/shexp                -> Q8_0 (blk < n_layer/8) else Q8_0_ROCMFPX
  ffn_gate_exps/shexp              -> Q8_0_ROCMFPX
  everything else quantizable      -> Q8_0_ROCMFPX

Usage:
    python convert-qwen35moe-rocmfp4.py <bf16-gguf> <out-gguf> \\
        [--output rocmfp4|strix_leon|rocmfpx] [nthreads]
"""

import sys
import os
import argparse
import ctypes
import numpy as np
import re
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'gguf-py'))
from gguf import GGUFReader, GGUFWriter, GGUFValueType
from gguf.constants import GGMLQuantizationType

# ---------------------------------------------------------------------------
# libggml-base loading (for quantization via direct ctypes calls)
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
            raise RuntimeError('libggml-base.so not found (run cmake --build first)')
    return _LIB


# ---------------------------------------------------------------------------
# Quantization functions using direct ctypes calls to libggml-base
# ---------------------------------------------------------------------------

def quantize_rocmfp4(data_f32):
    """Quantize flat f32 to Q4_0_ROCMFP4 (18 bytes per 32 elements)."""
    lib = _lib()
    fn = lib.rocmfp4_quantize_row_q4_0_ref
    fn.restype = None
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong]
    n = data_f32.size
    out = np.empty(n * 18 // 32, dtype=np.uint8)
    fn(data_f32.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p), n)
    return out


def quantize_rocmfp4_fast(data_f32):
    """Quantize flat f32 to Q4_0_ROCMFP4_FAST (17 bytes per 32 elements)."""
    lib = _lib()
    fn = lib.rocmfp4_quantize_row_q4_0_fast_ref
    fn.restype = None
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong]
    n = data_f32.size
    out = np.empty(n * 17 // 32, dtype=np.uint8)
    fn(data_f32.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p), n)
    return out


def quantize_q5k(data_f32):
    """Quantize flat f32 to Q5_K (176 bytes per 256 elements)."""
    lib = _lib()
    fn = lib.quantize_row_q5_K_ref
    fn.restype = None
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong]
    n = data_f32.size
    out = np.empty(n * 176 // 256, dtype=np.uint8)
    fn(data_f32.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p), n)
    return out


def quantize_q6k(data_f32):
    """Quantize flat f32 to Q6_K (210 bytes per 256 elements)."""
    lib = _lib()
    fn = lib.quantize_row_q6_K_ref
    fn.restype = None
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong]
    n = data_f32.size
    out = np.empty(n * 210 // 256, dtype=np.uint8)
    fn(data_f32.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p), n)
    return out


def quantize_rocmfpx(data_f32):
    """Quantize flat f32 to Q8_0_ROCMFPX (33 bytes per 32 elements)."""
    lib = _lib()
    fn = lib.rocmfpx_quantize_row_fp8_ref
    fn.restype = None
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong]
    n = data_f32.size
    out = np.empty(n * 33 // 32, dtype=np.uint8)
    fn(data_f32.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p), n)
    return out


def quantize_q8_0(data_f32):
    """Quantize flat f32 to Q8_0 (34 bytes per 32 elements)."""
    lib = _lib()
    fn = lib.quantize_row_q8_0_ref
    fn.restype = None
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_longlong]
    n = data_f32.size
    out = np.empty(n * 34 // 32, dtype=np.uint8)
    fn(data_f32.ctypes.data_as(ctypes.c_void_p), out.ctypes.data_as(ctypes.c_void_p), n)
    return out


# ---------------------------------------------------------------------------
# Read source tensor data as flat f32 (no full copy of memmap)
# ---------------------------------------------------------------------------
def read_source_data(src_ti):
    """Read source tensor data as a flat float32 array.

    Uses numpy.memmap view to avoid loading the entire file into RAM.
    For BF16, converts in-place via uint16 view without creating intermediate copies.
    """
    raw = np.array(src_ti.data)  # returns memmap for GGUF data section
    t = int(src_ti.tensor_type)
    if t in (GGMLQuantizationType.BF16, 30):
        u16 = raw.view(np.uint16)[: src_ti.n_elements * 2]
        return (u16.astype(np.uint32) << 16).view(np.float32).reshape(-1)
    return raw.astype(np.float32).reshape(-1)

# ---------------------------------------------------------------------------
# Quantization type routing
# ---------------------------------------------------------------------------
def parse_block_index(tensor_name):
    """Extract block index from tensor name, or None if not a block tensor."""
    m = re.match(r'blk\.(\d+)\.', tensor_name)
    return int(m.group(1)) if m else None


def use_more_bits(i_layer, n_layers):
    """Return True for layers that deserve extra quantization bits."""
    return (i_layer < n_layers // 8
            or i_layer >= 7 * n_layers // 8
            or (i_layer - n_layers // 8) % 3 == 2)


def classify_tensor(name):
    """Return a tensor category string for routing."""
    if 'output.weight' in name and 'blk.' not in name:
        return 'output'
    if 'token_embd.weight' in name:
        return 'token_embd'
    if 'attn_qkv.weight' in name:
        return 'attn_qkv'
    if 'attn_v.weight' in name:
        return 'attn_v'
    if 'attn_k.weight' in name:
        return 'attn_k'
    if 'attn_q.weight' in name:
        return 'attn_q'
    if 'attn_output.weight' in name:
        return 'attn_output'
    if 'ffn_down_exps.weight' in name or 'ffn_down_shexp.weight' in name:
        return 'ffn_down'
    if 'ffn_up_exps.weight' in name or 'ffn_up_shexp.weight' in name:
        return 'ffn_up'
    if 'ffn_gate_exps.weight' in name or 'ffn_gate_shexp.weight' in name:
        return 'ffn_gate'
    if 'ssm_alpha' in name or 'ssm_beta' in name or 'ssm_out.weight' in name:
        return 'ssm_fp4'
    if 'attn_gate.weight' in name:
        return 'attn_gate'
    return 'other'


def is_f32_only(name, shape):
    """Return True for tensors that must stay F32."""
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


def quant_type_for(name, shape, n_layer, output_mode):
    """Return target quant type string for this tensor.

    Args:
        name: tensor name
        shape: list of int dims
        n_layer: total number of blocks
        output_mode: 'rocmfp4', 'strix_leon', or 'rocmfpx'
    """
    cat = classify_tensor(name)
    blk = parse_block_index(name)
    i_layer = blk if blk is not None else 0

    # --- ROCmFP4 (file_type=100) ---
    if output_mode == 'rocmfp4':
        if cat == 'output':
            return 'Q6_K'
        if cat == 'attn_qkv':
            return 'Q5_K'
        if cat == 'attn_v':
            return 'Q5_K'
        if cat == 'ffn_down':
            # Q6_K for first 2/3 blocks, Q5_K for last 1/3
            if blk is not None and blk >= (2 * n_layer) // 3:
                return 'Q5_K'
            return 'Q6_K'
        if cat == 'ffn_gate':
            # Q5_K on "use_more_bits" layers, else default Q4_0_ROCMFP4
            if blk is not None and use_more_bits(blk, n_layer):
                return 'Q5_K'
            return 'Q4_0_ROCMFP4'
        if cat == 'ffn_up':
            return 'Q4_0_ROCMFP4'
        if cat in ('attn_k', 'attn_q', 'attn_output', 'attn_gate'):
            return 'Q4_0_ROCMFP4'
        if cat == 'ssm_fp4':
            return 'Q4_0_ROCMFP4'
        return 'Q4_0_ROCMFP4'

    # --- Strix Lean (file_type=106) ---
    if output_mode == 'strix_leon':
        if cat == 'token_embd':
            return 'Q5_K'
        if cat == 'output':
            return 'Q5_K'
        if cat == 'attn_qkv':
            return 'Q4_0_ROCMFP4'
        if cat in ('attn_v', 'attn_k'):
            return 'Q4_0_ROCMFP4'
        if cat in ('ffn_down', 'ffn_up', 'ffn_gate'):
            return 'Q4_0_ROCMFP4_FAST'
        if cat in ('attn_q', 'attn_output', 'attn_gate'):
            return 'Q4_0_ROCMFP4_FAST'
        if cat == 'ssm_fp4':
            return 'Q4_0_ROCMFP4_FAST'
        return 'Q4_0_ROCMFP4_FAST'

    # --- Q8_0_ROCMFPX (file_type=111) ---
    if output_mode == 'rocmfpx':
        if cat == 'token_embd':
            return 'Q8_0_ROCMFPX'
        if cat == 'output':
            return 'Q8_0_ROCMFPX'
        if cat == 'attn_qkv':
            return 'Q8_0'
        if cat in ('attn_v', 'attn_k', 'attn_q', 'attn_output'):
            return 'Q8_0_ROCMFPX'
        if cat == 'ffn_down':
            # Boost to Q8_0 for first 1/8 and last 1/4 of layers
            if blk is not None and (blk < n_layer // 8 or blk >= 3 * n_layer // 4):
                return 'Q8_0'
            return 'Q8_0_ROCMFPX'
        if cat == 'ffn_up':
            # Boost to Q8_0 for first 1/8 of layers
            if blk is not None and blk < n_layer // 8:
                return 'Q8_0'
            return 'Q8_0_ROCMFPX'
        if cat == 'ffn_gate':
            return 'Q8_0_ROCMFPX'
        if cat == 'ssm_fp4':
            return 'Q8_0_ROCMFPX'
        if cat == 'attn_gate':
            return 'Q8_0_ROCMFPX'
        return 'Q8_0_ROCMFPX'

    raise ValueError(f'Unknown output_mode: {output_mode!r}')


# Mapping from quant type string to quantization function
QUANT_FUNCS = {
    'Q4_0_ROCMFP4':      quantize_rocmfp4,
    'Q4_0_ROCMFP4_FAST': quantize_rocmfp4_fast,
    'Q5_K':              quantize_q5k,
    'Q6_K':              quantize_q6k,
    'Q8_0_ROCMFPX':      quantize_rocmfpx,
    'Q8_0':              quantize_q8_0,
}

# Mapping from quant type string to raw GGUF dtype and byte-shape divisor info
QUANT_META = {
    'Q4_0_ROCMFP4':      (GGMLQuantizationType.Q4_0_ROCMFP4,     18, 32),
    'Q4_0_ROCMFP4_FAST': (GGMLQuantizationType.Q4_0_ROCMFP4_FAST, 17, 32),
    'Q5_K':              (GGMLQuantizationType.Q5_K,              176, 256),
    'Q6_K':              (GGMLQuantizationType.Q6_K,              210, 256),
    'Q8_0_ROCMFPX':      (GGMLQuantizationType.Q8_0_ROCMFPX,     33, 32),
    'Q8_0':              (GGMLQuantizationType.Q8_0,             34, 32),
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Convert Qwen35 MoE/Dense bf16 GGUF to ROCmFP4 / Q8_0_ROCMFPX')
    parser.add_argument('bf16_gguf', help='source bf16 GGUF')
    parser.add_argument('out_gguf', help='output quantized GGUF')
    parser.add_argument('--output', '-o',
                        choices=['rocmfp4', 'strix_leon', 'rocmfpx'],
                        default='rocmfp4',
                        help='quantization recipe (default: rocmfp4)')
    parser.add_argument('--threads', '-t', type=int, default=4,
                        help='quantize thread count (default: 4)')
    args = parser.parse_args()

    BF16_GGUF = args.bf16_gguf
    OUT_GGUF = args.out_gguf
    N_THREADS = args.threads
    OUTPUT_MODE = args.output

    r = GGUFReader(BF16_GGUF, 'r')
    print(f'Reading {BF16_GGUF} ({len(r.tensors)} tensors)')

    # Read model metadata
    n_layer = 0
    for k, v in r.fields.items():
        if k == 'qwen35moe.block_count':
            n_layer = int(v.contents())
        elif k == 'qwen35.block_count':
            n_layer = int(v.contents())
        elif k == 'general.architecture':
            arch = v.contents()
            print(f'Architecture: {arch}')

    if n_layer == 0:
        # Fallback: count blocks from tensor names
        block_set = set()
        for t in r.tensors:
            m = re.match(r'blk\.(\d+)\.', t.name)
            if m:
                block_set.add(int(m.group(1)))
        n_layer = max(block_set) + 1 if block_set else 40
        print(f'Detected n_layer={n_layer} from tensor names')

    # Categorize tensors
    tensors_info = []
    for ti in r.tensors:
        name = ti.name
        shape = [int(x) for x in ti.shape]
        if is_f32_only(name, shape):
            tensors_info.append((name, shape, False, 'F32'))
            continue
        tensors_info.append((name, shape, True,
                             quant_type_for(name, shape, n_layer, OUTPUT_MODE)))

    # Print type distribution
    type_counts = Counter(t[3] for t in tensors_info)
    print(f'\nQuantization type distribution ({OUTPUT_MODE}):')
    for k, v in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f'  {k}: {v}')

    # Determine output file_type
    file_types = {
        'rocmfp4':    (100, 'Q4_0_ROCMFP4'),
        'strix_leon': (106, 'Q4_0_ROCMFP4_STRIX_LEAN'),
        'rocmfpx':    (111, 'Q8_0_ROCMFPX'),
    }
    OUT_FILE_TYPE, OUT_LABEL = file_types[OUTPUT_MODE]
    print(f'Output file_type={OUT_FILE_TYPE} ({OUT_LABEL})')

    # Write GGUF
    w = GGUFWriter(OUT_GGUF, 'qwen35moe')

    # Copy metadata
    for key, f in r.fields.items():
        if key in ('GGUF.version', 'GGUF.tensor_count', 'GGUF.kv_count'):
            continue
        if key == 'general.architecture':
            # Keep the original architecture value
            arch_val = f.contents()
            w.add_key_value(key, arch_val, GGUFValueType.STRING)
            continue
        if key == 'general.file_type':
            w.add_key_value(key, OUT_FILE_TYPE, GGUFValueType.UINT32)
            continue
        t = f.types
        vtype = t[0]
        if vtype == GGUFValueType.ARRAY:
            w.add_key_value(key, f.contents(), vtype, sub_type=t[1])
        else:
            val = f.contents()
            w.add_key_value(key,
                            val.tolist() if isinstance(val, np.ndarray) else val,
                            vtype)
    print('Metadata copied')

    # Build lookup
    src_by_name = {ti.name: ti for ti in r.tensors}

    def quantize_job(job):
        name, shape, _, qtype = job
        data = read_source_data(src_by_name[name])
        result = QUANT_FUNCS[qtype](data)
        return name, result, qtype

    quant_jobs = [t for t in tensors_info if t[2]]
    results = {}
    print(f'\nQuantizing {len(quant_jobs)} tensors...')
    with ThreadPoolExecutor(max_workers=max(1, N_THREADS)) as pool:
        for idx, res in enumerate(pool.map(quantize_job, quant_jobs), 1):
            name, result, qtype = res
            results[name] = (result, qtype)
            if idx % 50 == 0:
                print(f'  quantized {idx}/{len(quant_jobs)} tensors')

    # Write tensors
    written = 0
    for name, shape, should_quantize, qtype in tensors_info:
        orig_shape = list(map(int, shape))
        if not should_quantize or qtype == 'F32' or name not in results:
            data = read_source_data(src_by_name[name])
            w.add_tensor(name, data,
                         raw_shape=list(reversed(orig_shape)),
                         raw_dtype=GGMLQuantizationType.F32)
        else:
            quantized, _ = results[name]
            # Build byte shape for the quantized tensor
            # For 2D+ tensors: byte_shape = [n_per_row, n_rows * type_size / blk_size]
            # where type_size/blk_size is bytes per element in the quantized format
            raw_dtype, type_size, blk_size = QUANT_META[qtype]
            if len(orig_shape) >= 2:
                n_per_row = orig_shape[1]
                byte_shape = [n_per_row, orig_shape[0] * type_size // blk_size]
            else:
                byte_shape = [orig_shape[0] * type_size // blk_size]
            w.add_tensor(name, quantized,
                         raw_shape=byte_shape,
                         raw_dtype=raw_dtype)
        written += 1
        if written % 50 == 0:
            print(f'  {written}/{len(tensors_info)} tensors written')

    assert written == len(r.tensors)

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    size_gb = os.path.getsize(OUT_GGUF) / 1e9
    print(f'\nDone: {OUT_GGUF} ({size_gb:.2f} GB, {written} tensors)')


if __name__ == '__main__':
    main()
