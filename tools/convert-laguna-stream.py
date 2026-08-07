#!/usr/bin/env python3
"""
Streaming safetensors -> ROCmFP4 GGUF for Laguna-S-2.1.

No BF16 intermediate file: every tensor is transposed, quantized to ROCmFP4
(the fork's AMD-tuned 4-bit format, 18 bytes / 32 values, exhaustive UE4M3
scale search) and written to the GGUF data section immediately. Quantization
calls the C reference (rocmfp4_quantize_row_q4_0_ref via ctypes) so the output
matches llama-quantize byte for byte. Memory stays O(one tensor).

Layout ground truth: /opt/LLM/hf/jcbtc/Laguna-S-2.1-NVFP4.gguf (ModelOpt build)
minus its e8m0-specific scale/input_scale tensors == 814 tensors. The ModelOpt
file's e8m0 scale packing is incompatible with llama.cpp, hence the
re-quantization from BF16 safetensors (ROCmFP4 beats NVFP4 on gfx906 in both
precision and speed; NVFP4 is kept only for file compatibility).

Tensors that must NOT be quantized (row length not a multiple of the 32-value
ROCmFP4 block, or 1D convention):
  - all 1D norms / biases / q_norm / k_norm / exp_probs_b  -> F32 (as in reference)
  - attn_gate (48/72 heads -> rows of 48/72, ggml_row_size asserts 32-alignment) -> F32
  - ffn_gate_inp (router, reference keeps F32 to protect top-k selection) -> F32

Usage: python convert-laguna-stream.py <shell-gguf> <src-dir> <out-gguf>
<shell-gguf> is a metadata-only GGUF from make-laguna-shell.py (no tensors)
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

from rocmfp4_quant import quantize_rocmfp4

# ---------------- metadata copy ----------------
w = GGUFWriter(OUT, 'laguna')
HEADER_FIELDS = {'GGUF.version', 'GGUF.tensor_count', 'GGUF.kv_count'}  # synthetic, from reader header
for key, f in r.fields.items():
    if key in HEADER_FIELDS:
        continue  # header fields, not real kv
    t = f.types
    vtype = t[0]
    if key == 'general.file_type':
        # tensors are ROCmFP4 now
        w.add_key_value(key, 100, vtype)  # GGML_FTYPE_MOSTLY_Q4_0_ROCMFP4
        continue
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
        return 'token_embd.weight'
    if key == 'lm_head.weight':
        return 'output.weight'
    if key == 'model.norm.weight':
        return 'output_norm.weight'
    il = int(re.match(r'^model\.layers\.(\d+)\.', key).group(1))
    if 'shared_expert' in key:
        kind = re.search(r'\.(gate|up|down)_proj\.weight$', key).group(1)
        return f'blk.{il}.ffn_{kind}_shexp.weight'
    kind = RE_KIND.search(key).group(1)
    if kind == 'input_layernorm':    return f'blk.{il}.attn_norm.weight'
    if kind == 'post_attention_layernorm': return f'blk.{il}.ffn_norm.weight'
    if kind == 'q_proj':             return f'blk.{il}.attn_q.weight'
    if kind == 'k_proj':             return f'blk.{il}.attn_k.weight'
    if kind == 'v_proj':             return f'blk.{il}.attn_v.weight'
    if kind == 'o_proj':             return f'blk.{il}.attn_output.weight'
    if kind == 'q_norm':             return f'blk.{il}.attn_q_norm.weight'
    if kind == 'k_norm':             return f'blk.{il}.attn_k_norm.weight'
    if kind == 'g_proj':             return f'blk.{il}.attn_gate.weight'
    if kind == 'gate':               return f'blk.{il}.ffn_gate_inp.weight'
    if kind == 'gate_proj':          return f'blk.{il}.ffn_gate.weight'
    if kind == 'up_proj':            return f'blk.{il}.ffn_up.weight'
    if kind == 'down_proj':          return f'blk.{il}.ffn_down.weight'
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
                    # numpy shape (n_expert, n_ff, n_embd); GGUFWriter reverses
                    # to on-disk dims (n_embd, n_ff, n_expert) == llama.cpp's
                    # create_tensor order
                    tensors[out] = [[N_EXPERT, shp[0], shp[1]], str(shard), key]
                exp_keys.setdefault((il, kind), []).append((str(shard), key))
                continue
            if RE_BIAS.match(key):
                il = int(RE_BIAS.match(key).group(1))
                tensors[f'blk.{il}.exp_probs_b.bias'] = [list(shp), str(shard), key]
                continue
            if RE_FLAT.match(key):
                # HF safetensors layout == GGUF numpy layout (row-major, last
                # dim fastest); the writer emits on-disk dims reversed, which
                # is what llama.cpp's create_tensor expects. No transpose.
                tensors[gguf_name(key)] = [list(shp), str(shard), key]
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
        assert shape[-1] % 32 == 0, f'{name}: row {shape[-1]} not ROCmFP4-block aligned'
        byte_shape = shape[:-1] + [shape[-1] * 18 // 32]
        w.add_tensor_info(name, byte_shape, np.uint8, n * 18 // 32,
                          raw_dtype=GGMLQuantizationType.Q4_0_ROCMFP4)

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
        q = quantize_rocmfp4(t.contiguous().to(torch.float32))
        del t
    elif tensor_is_f32(name, shape):
        t = open_shards[shard].get_slice(key)[:]
        assert t.shape == tuple(shape), f'{name}: {t.shape} vs {shape}'
        q = t.contiguous().to(torch.float32)
        del t
    else:
        t = open_shards[shard].get_slice(key)[:]
        assert t.shape == tuple(shape), f'{name}: {t.shape} vs {shape}'
        q = quantize_rocmfp4(t.contiguous().to(torch.float32))
        del t
    w.write_tensor_data(q.cpu().numpy())
    del q
    written += 1
    if written % 100 == 0:
        print(f'  {written}/{len(order)} tensors written')

assert written == len(order), f'{written} != {len(order)}'
# note: safetensors' safe_open (Rust builtin since 0.8.0) has no close();
# handles are released at process exit
w.close()
import os
print(f'done: {OUT} ({os.path.getsize(OUT)/1e9:.2f} GB, {written} tensors)')
