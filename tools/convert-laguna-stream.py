#!/usr/bin/env python3
"""
Streaming safetensors -> NVFP4 GGUF for Laguna-S-2.1.

No BF16 intermediate file: every tensor is transposed, quantized to NVFP4
(torch, bit-identical semantics to ggml-quants.c quantize_row_nvfp4_ref) and
written to the GGUF data section immediately. Memory stays O(one tensor).

Layout ground truth: /opt/LLM/hf/jcbtc/Laguna-S-2.1-NVFP4.gguf (ModelOpt build)
minus its e8m0-specific scale/input_scale tensors == 814 tensors. The ModelOpt
file runs on this fork (shapes verified) but its scales are e8m0 (NVIDIA
packing) while this fork's GGML_TYPE_NVFP4 (36 bytes / 64 values) uses e4m3
scales, hence the re-quantization from BF16 safetensors.

Tensors that must NOT be quantized (row length not a multiple of the 64-value
NVFP4 block, or 1D convention):
  - all 1D norms / biases / q_norm / k_norm / exp_probs_b  -> F32 (as in reference)
  - attn_gate (48/72 heads -> rows of 48/72, ggml_row_size asserts 64-alignment) -> F32
  - ffn_gate_inp (router, reference keeps F32 to protect top-k selection) -> F32

Usage: python convert-laguna-stream.py <src-gguf> <src-dir> <out-gguf>
"""

import re
import sys
import numpy as np
import torch
from pathlib import Path
from safetensors.torch import load_file  # noqa: F401  (imported for side effects)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'gguf-py'))
from gguf import GGUFReader, GGUFWriter, GGUFValueType, GGMLQuantizationType

SRC_GGUF, SRC_DIR, OUT = sys.argv[1], Path(sys.argv[2]), sys.argv[3]

r = GGUFReader(SRC_GGUF, 'r')

N_LAYER, N_EXPERT, SWA_PERIOD = 48, 256, 4
N_HEADS_FULL, N_HEADS_SWA = 48, 72

def layer_heads(il):
    return N_HEADS_FULL if il % SWA_PERIOD == 0 else N_HEADS_SWA

from nvfp4_quant import quantize_nvfp4

# ---------------- metadata copy ----------------
w = GGUFWriter(OUT, 'laguna')
HEADER_FIELDS = {'GGUF.version', 'GGUF.tensor_count', 'GGUF.kv_count'}  # synthetic, from reader header
for key, f in r.fields.items():
    if key in HEADER_FIELDS:
        continue  # header fields, not real kv
    t = f.types
    vtype = t[0]
    if key == 'laguna.attention.head_count':
        # already an array in the source; re-emit defensively (element type INT32)
        w.add_key_value(key, [layer_heads(i) for i in range(N_LAYER)],
                        GGUFValueType.ARRAY, sub_type=GGUFValueType.INT32)
        continue
    if vtype == GGUFValueType.ARRAY:
        w.add_key_value(key, f.contents(), vtype, sub_type=t[1])
    else:
        val = f.contents()
        w.add_key_value(key, val.tolist() if isinstance(val, np.ndarray) else val, vtype)
print('metadata copied')

# ---------------- pass 1: register tensor info (no data loaded) ----------------
RE_EXP   = re.compile(r'^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$')
RE_BIAS  = re.compile(r'^model\.layers\.(\d+)\.mlp\.experts\.e_score_correction_bias$')
RE_FLAT  = re.compile(r'^(?:model\.(?:embed_tokens|norm)\.weight|lm_head\.weight|'
                      r'model\.layers\.(\d+)\.(?:input_layernorm|post_attention_layernorm|'
                      r'self_attn\.(?:q_proj|k_proj|v_proj|o_proj|q_norm|k_norm|g_proj)|'
                      r'mlp\.(?:gate|gate_proj|up_proj|down_proj)|'
                      r'mlp\.shared_expert\.(?:gate_proj|up_proj|down_proj))\.weight)$')
RE_KIND  = re.compile(r'\.(input_layernorm|post_attention_layernorm|q_proj|k_proj|v_proj|o_proj|q_norm|k_norm|g_proj|gate|gate_proj|up_proj|down_proj)\.weight$')

SHARDS = sorted(SRC_DIR.glob('model-*.safetensors'))

def gguf_name(key):
    if key == 'model.embed_tokens.weight':
        return 'token_embd.weight', True
    if key == 'lm_head.weight':
        return 'output.weight', True
    if key == 'model.norm.weight':
        return 'output_norm.weight', False
    il = int(re.match(r'^model\.layers\.(\d+)\.', key).group(1))
    if 'shared_expert' in key:
        kind = re.search(r'\.(gate|up|down)_proj\.weight$', key).group(1)
        return f'blk.{il}.ffn_{kind}_shexp.weight', True
    kind = RE_KIND.search(key).group(1)
    if kind == 'input_layernorm':    return f'blk.{il}.attn_norm.weight', False
    if kind == 'post_attention_layernorm': return f'blk.{il}.ffn_norm.weight', False
    if kind == 'q_proj':             return f'blk.{il}.attn_q.weight', True
    if kind == 'k_proj':             return f'blk.{il}.attn_k.weight', True
    if kind == 'v_proj':             return f'blk.{il}.attn_v.weight', True
    if kind == 'o_proj':             return f'blk.{il}.attn_output.weight', True
    if kind == 'q_norm':             return f'blk.{il}.attn_q_norm.weight', False
    if kind == 'k_norm':             return f'blk.{il}.attn_k_norm.weight', False
    if kind == 'g_proj':             return f'blk.{il}.attn_gate.weight', True
    if kind == 'gate':               return f'blk.{il}.ffn_gate_inp.weight', True
    if kind == 'gate_proj':          return f'blk.{il}.ffn_gate.weight', True
    if kind == 'up_proj':            return f'blk.{il}.ffn_up.weight', True
    if kind == 'down_proj':          return f'blk.{il}.ffn_down.weight', True
    raise KeyError(kind)

# collect tensor shapes from safetensors metadata (no data)
from safetensors import safe_open
tensors = {}  # gguf name -> [shape_in_gguf, shard, src_key]
exp_keys = {}  # (il, kind) -> [(shard, src_key), ...]
for shard in SHARDS:
    with safe_open(str(shard), framework='pt') as f:
        for key in f.keys():
            shp = tuple(f.get_slice(key).get_shape())
            m = RE_EXP.match(key)
            if m:
                il, e, kind = int(m.group(1)), int(m.group(2)), m.group(3)
                out = f'blk.{il}.ffn_{"down" if kind == "down_proj" else ("gate" if kind == "gate_proj" else "up")}_exps.weight'
                if out not in tensors:
                    tensors[out] = [[shp[1], shp[0], N_EXPERT], str(shard), key]
                exp_keys.setdefault((il, kind), []).append((str(shard), key))
                continue
            if RE_BIAS.match(key):
                il = int(RE_BIAS.match(key).group(1))
                tensors[f'blk.{il}.exp_probs_b.bias'] = [list(shp), str(shard), key]
                continue
            if RE_FLAT.match(key):
                name, tr = gguf_name(key)
                tensors[name] = [list(shp[::-1]) if tr else list(shp), str(shard), key]
                continue
            raise KeyError(f'unmapped {key}')
    print(f'  pass1 {shard.name}: {len(tensors)} tensors')

# tensor order: fixed, matching metadata (sorted by name keeps it deterministic)
order = sorted(tensors)
assert len(order) == 814, f'expected 814 tensors (reference minus e8m0 scales), got {len(order)}'

def tensor_is_f32(name, shape):
    return len(shape) == 1 or name.endswith('attn_gate.weight') or name.endswith('ffn_gate_inp.weight')

for name in order:
    shape = tensors[name][0]
    n = 1
    for s in shape:
        n *= s
    if tensor_is_f32(name, shape):
        w.add_tensor_info(name, shape, np.float32, n * 4)
    else:
        assert shape[-1] % 64 == 0, f'{name}: row {shape[-1]} not NVFP4-block aligned'
        byte_shape = shape[:-1] + [shape[-1] * 36 // 64]
        w.add_tensor_info(name, byte_shape, np.uint8, n * 36 // 64,
                          raw_dtype=GGMLQuantizationType.NVFP4)

w.write_header_to_file()
w.write_kv_data_to_file()
w.write_ti_data_to_file()
print(f'header+kv+ti written, {len(order)} tensors registered')

# ---------------- pass 2: stream data, in info order ----------------
written = 0
open_shards = {}
for name in order:
    _, shard, key = tensors[name]
    shape = tensors[name][0]
    if shard not in open_shards:
        open_shards[shard] = safe_open(shard, framework='pt')
    if name.endswith('_exps.weight'):
        # gather all 256 experts of this layer/kind and stack along a NEW leading
        # dim: memory order (expert, out, in) == GGUF row-major (ne0=in fastest)
        il = int(name.split('.')[1])
        kind = 'gate_proj' if 'gate_exps' in name else ('up_proj' if 'up_exps' in name else 'down_proj')
        keys = exp_keys[(il, kind)]
        assert len(keys) == N_EXPERT, f'{name}: {len(keys)} experts'
        parts = []
        for s, k in sorted(keys, key=lambda x: int(x[1].split('.experts.')[1].split('.')[0])):
            if s not in open_shards:
                open_shards[s] = safe_open(s, framework='pt')
            parts.append(open_shards[s].get_slice(k)[:])
        t = torch.stack(parts, dim=0)  # [256, out, in]
        del parts
        q = quantize_nvfp4(t.contiguous().to(torch.float32))
        del t
    elif tensor_is_f32(name, shape):
        t = open_shards[shard].get_slice(key)[:]
        if tuple(t.shape) != tuple(shape):
            t = t.T
        assert t.shape == tuple(shape), f'{name}: {t.shape} vs {shape}'
        q = t.contiguous().to(torch.float32)
        del t
    else:
        t = open_shards[shard].get_slice(key)[:]
        if tuple(t.shape) != tuple(shape):
            t = t.T
        assert t.shape == tuple(shape), f'{name}: {t.shape} vs {shape}'
        q = quantize_nvfp4(t.contiguous().to(torch.float32))
        del t
    w.write_tensor_data(q.cpu().numpy())
    del q
    written += 1
    if written % 100 == 0:
        print(f'  {written}/{len(order)} tensors written')
for f in open_shards.values():
    f.close()

assert written == len(order), f'{written} != {len(order)}'
w.close()
import os
print(f'done: {OUT} ({os.path.getsize(OUT)/1e9:.2f} GB, {written} tensors)')
