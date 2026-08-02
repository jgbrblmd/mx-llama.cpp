#!/usr/bin/env python3
"""
Convert Laguna-S-2.1 safetensors (BF16) to a BF16 GGUF.

The downloaded "NVFP4" GGUF uses a scale layout (e8m0) incompatible with
llama.cpp's NVFP4 (ue4m3). Rebuild from the original weights instead:
  * metadata + tokenizer are copied verbatim from the NVFP4 GGUF
  * laguna.attention.head_count is rewritten as a per-layer array
    (48 for full-attention layers, 72 for sliding-window layers)
  * weights are streamed from safetensors shards (transposed [out,in] -> [in,out])
  * MoE experts are stacked into [in, out, n_expert] 3D tensors

Usage: python convert-laguna-safetensors.py <src-gguf> <src-dir> <out-gguf>
"""

import re
import sys
import numpy as np
import torch
from pathlib import Path
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'gguf-py'))
from gguf import GGUFReader, GGUFWriter, GGUFValueType, GGMLQuantizationType

SRC_GGUF, SRC_DIR, OUT = sys.argv[1], Path(sys.argv[2]), sys.argv[3]

r = GGUFReader(SRC_GGUF, 'r')

# ---- layer structure from config ----
N_LAYER    = 48
N_EXPERT   = 256
SWA_PERIOD = 4          # full at il % 4 == 0
N_HEADS_FULL, N_HEADS_SWA = 48, 72

def layer_heads(il: int) -> int:
    return N_HEADS_FULL if il % SWA_PERIOD == 0 else N_HEADS_SWA

# ---- reference shapes from the source GGUF (sanity check) ----
ref_shape = {t.name: tuple(t.shape) for t in r.tensors}

# ---- 1. copy all metadata ----
w = GGUFWriter(OUT, 'laguna', use_temp_file=True)

n_kv = 0
for key, f in r.fields.items():
    t = f.types
    vtype = t[0]
    if key == 'laguna.attention.head_count':
        # per-layer array: [48,72,72,72] x 12
        w.add_array(key, [layer_heads(i) for i in range(N_LAYER)])
        n_kv += 1
        continue
    if key == 'general.file_type':
        # weights are BF16 now, not NVFP4
        w.add_key_value(key, 32, vtype)  # GGML_FTYPE_MOSTLY_BF16
        n_kv += 1
        continue
    if vtype == GGUFValueType.ARRAY:
        sub = t[1]
        vals = f.contents()
        w.add_key_value(key, vals, vtype, sub_type=sub)
    else:
        val = f.contents()
        if isinstance(val, np.ndarray):
            val = val.tolist()
        w.add_key_value(key, val, vtype)
    n_kv += 1
print(f'metadata copied: {n_kv} keys')

# ---- 2. tensor mapping ----
SHARDS = sorted(SRC_DIR.glob('model-*.safetensors'))
print(f'{len(SHARDS)} shards')

# regexes
RE_EXP   = re.compile(r'^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$')
RE_BIAS  = re.compile(r'^model\.layers\.(\d+)\.mlp\.experts\.e_score_correction_bias$')
RE_FLAT  = re.compile(r'^(?:model\.(?:embed_tokens|norm)\.weight|lm_head\.weight|'
                      r'model\.layers\.(\d+)\.(?:input_layernorm|post_attention_layernorm|'
                      r'self_attn\.(?:q_proj|k_proj|v_proj|o_proj|q_norm|k_norm|g_proj)|'
                      r'mlp\.(?:gate|gate_proj|up_proj|down_proj)|'
                      r'mlp\.shared_expert\.(?:gate_proj|up_proj|down_proj))\.weight)$')
RE_KIND  = re.compile(r'\.(input_layernorm|post_attention_layernorm|q_proj|k_proj|v_proj|o_proj|q_norm|k_norm|g_proj|gate|gate_proj|up_proj|down_proj)\.weight$')

def gguf_name(m):
    key = m.group(0)
    if key == 'model.embed_tokens.weight':
        return 'token_embd.weight', True
    if key == 'lm_head.weight':
        return 'output.weight', True
    if key == 'model.norm.weight':
        return 'output_norm.weight', False
    i = int(m.group(1))
    kind = RE_KIND.search(key).group(1)
    if kind == 'input_layernorm':
        return f'blk.{i}.attn_norm.weight', False
    if kind == 'post_attention_layernorm':
        return f'blk.{i}.ffn_norm.weight', False
    if kind == 'q_proj':
        return f'blk.{i}.attn_q.weight', True
    if kind == 'k_proj':
        return f'blk.{i}.attn_k.weight', True
    if kind == 'v_proj':
        return f'blk.{i}.attn_v.weight', True
    if kind == 'o_proj':
        return f'blk.{i}.attn_output.weight', True
    if kind == 'q_norm':
        return f'blk.{i}.attn_q_norm.weight', False
    if kind == 'k_norm':
        return f'blk.{i}.attn_k_norm.weight', False
    if kind == 'g_proj':
        return f'blk.{i}.attn_gate.weight', True
    # mlp
    if kind == 'gate':
        return f'blk.{i}.ffn_gate_inp.weight', True
    if kind == 'gate_proj':
        return f'blk.{i}.ffn_gate.weight', True
    if kind == 'up_proj':
        return f'blk.{i}.ffn_up.weight', True
    if kind == 'down_proj':
        return f'blk.{i}.ffn_down.weight', True
    raise KeyError(kind)

def bf16_bits(t: torch.Tensor) -> np.ndarray:
    """safetensors tensor (bf16) -> uint16 bit-pattern numpy array."""
    t = t.contiguous()
    return t.view(torch.uint16).cpu().numpy()

def add_t(name: str, t: torch.Tensor, transpose: bool):
    if transpose:
        t = t.T
    shape = tuple(t.shape)
    if name in ref_shape:
        assert shape == ref_shape[name], f'{name}: shape {shape} != ref {ref_shape[name]}'
    w.add_tensor(name, bf16_bits(t), raw_dtype=GGMLQuantizationType.BF16)

n_added = 0
exp_buf = {}  # il -> {kind -> {expert_id: tensor}}; stacked once complete
for shard in SHARDS:
    st = load_file(str(shard))
    # MoE experts of this shard, accumulated across shards per layer
    for key in list(st.keys()):
        m = RE_EXP.match(key)
        if m:
            il, e, kind = int(m.group(1)), int(m.group(2)), m.group(3)
            exp_buf.setdefault(il, {}).setdefault(kind, {})[e] = st.pop(key)
            continue
        m = RE_BIAS.match(key)
        if m:
            il = int(m.group(1))
            add_t(f'blk.{il}.ffn_exp_probs_b.bias', st.pop(key), False)
            n_added += 1
            continue
        m = RE_FLAT.match(key)
        if not m:
            print(f'  WARN: unmapped key {key}')
            continue
        name, tr = gguf_name(m)
        add_t(name, st.pop(key), tr)
        n_added += 1
    # stack complete layers (all N_EXPERT experts of a kind present)
    for il in sorted(exp_buf):
        kinds = exp_buf[il]
        done = True
        for kind, emap in kinds.items():
            if len(emap) < N_EXPERT:
                done = False
                break
        if not done:
            continue
        for kind, emap in sorted(kinds.items()):
            assert sorted(emap) == list(range(N_EXPERT)), f'layer {il} {kind}: missing experts'
            # [out, in] -> [in, out, n_expert]
            stacked = torch.stack([emap[e].T for e in range(N_EXPERT)], dim=-1)
            name = f'blk.{il}.ffn_{"down" if kind == "down_proj" else ("gate" if kind == "gate_proj" else "up")}_exps.weight'
            add_t(name, stacked, transpose=False)
            n_added += 1
        del exp_buf[il]
    del st
    print(f'  shard {shard.name}: done ({n_added} tensors so far, {len(exp_buf)} layers buffered)')

assert not exp_buf, f'layers never completed: {sorted(exp_buf)}'

print(f'total tensors: {n_added}')
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_tensors_to_file()
w.close()
print(f'written: {OUT}')
