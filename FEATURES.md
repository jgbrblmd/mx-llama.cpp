# Features

This fork extends upstream llama.cpp with multi-GPU and speculative-decoding
optimizations. Most additions are backend-generic; the gfx906 (VEGA20) kernel
tuning is the only hardware-specific part.

## Multi-stage tensor parallelism (`-tps`)

Upstream's `-sm tensor` runs every layer as one tensor-parallel group across all
GPUs. This fork adds `-tps T` (`--tensor-parallel-size`): split the GPUs into
groups of T, tensor-parallel within each group, and pipeline the layers across
the groups. `T=0` (default) preserves upstream's single-group behavior; `T>0`
requires `n_gpus % T == 0`. Backend-generic.

## Custom GPU AllReduce

An optional peer-write broadcast plus two-shot reduce-scatter / allgather
AllReduce for the tensor-parallel reduction (in addition to upstream's
`allreduce.cu`). F32 on the wire and faster than the RCCL / NCCL ring for token
generation over PCIe. Enable with `GGML_ENABLE_CUSTOM_AR=1`; the fast peer-write
path needs fine-grain PCIe coherence (`HSA_FORCE_FINE_GRAIN_PCIE=1` on any AMD
over PCIe, a no-op on hardware-coherent GPUs and ignored on NVIDIA). Validated
on gfx906.

## MTP speculative-decode optimizations

Opt-in optimizations on top of upstream's `--spec-type draft-mtp` (MTP and the
Qwen3.6 head are upstream), enabled with `LLAMA_ENABLE_MTP_OPT=1`: deferred-prefill
KV staging, a KV-only prefill replay, disabling the draft context's pipeline ring,
and a non-finite-draft fail-safe. Default off uses the standard `draft-mtp` path
with these disabled. Backend-generic.

## Concurrent lane dispatch

Under `-sm tensor` the meta backend issued each subgraph to its GPUs in device
order, so the AllReduce closing it waited on the last one, and with 80 to 130
such subgraphs per token depending on the model, that stagger was rebuilt at
every one. The lanes are now
issued concurrently, which is bit-exact. The gain tracks how many GPUs share one
tensor-parallel group: measured on Qwen3.6-35B-A3B, +32% token generation on an
8-GPU tensor split and +2.5% on 4, with prefill flat. Under multi-stage `-tps`
it follows the group size rather than the total GPU count. On by default;
`GGML_META_PARALLEL_DISPATCH=0` restores the serial issue. Inert unless at least
two GPUs share a tensor split. CUDA / ROCm sub-backends only.

## Whole-token graph capture

A decode token under `-sm tensor` made one host round trip per AllReduce-bounded
subgraph (80 to 130 per token depending on the model), and the collective billed
the host submission spread at each. Each GPU now records its whole token (subgraph, AllReduce, subgraph, and
so on) into a single CUDA or HIP graph and replays that once per token, which is
bit-exact. Worth +6% token generation on a 4-GPU tensor split, on both a MoE and
a dense model. On 8 GPUs the throughput gain is small but run-to-run spread
drops from 11% to 2.5%. Prefill is unchanged by design. On by default;
`GGML_META_TOKEN_GRAPH=0` restores the per-subgraph dispatch. Requires the
concurrent lane dispatch above. Validated on gfx906.

## Multi-GPU transfer tuning

Hardware-queue handling (`GPU_MAX_HW_QUEUES`) and an optional RCCL point-to-point
stage-transfer path (`GGML_META_XFER_RCCL`) for the multi-stage pipeline.

## gfx906 kernel tuning

Hardware-specific tuning for gfx906 / VEGA20 (MI50, MI60, Radeon VII, Radeon Pro
VII): MMQ tile-width selection, q8_1 quantization, top-k MoE row handling, and
gated-delta-net warp counts.

## BF16 compute on AMD without native bfloat16

On AMD parts predating CDNA and RDNA3, BF16 matmuls compute in F32. rocBLAS has
no tuned bf16 kernel for that hardware, and `compute_type=BF16` also rounds the
F32 activations down to bf16, so F32 is both faster and more faithful. Automatic,
no flag. Worth +18-19% prefill on UD / `*_XL` quants that keep BF16 tensors.
`GGML_CUDA_CUBLAS_COMPUTE_TYPE=bf16` selects the old compute type.

## Quantized activation reuse

Several matmuls usually read one activation (q/k/v off a single attn_norm, the
router and gate/up off a single ffn_norm), and each quantized it to q8_1 again.
The quantized copy is now kept and handed to the later matmuls, which is
bit-exact. Worth +2.2-2.6% on prefill and decode. On by default;
`GGML_CUDA_Q8_1_CACHE=0` restores the old behavior. Backend-generic.

## Building from source

Requires a ROCm toolchain with gfx906 support (rocBLAS gfx906 kernels, plus RCCL
for `GGML_HIP_RCCL`). Note gfx906 is deprecated in ROCm 7.x. See `docs/build.md`
for general HIP build background.

```bash
cmake -B build \
  -DGGML_HIP=ON \
  -DGGML_HIP_GRAPHS=ON \
  -DGGML_HIP_RCCL=ON \
  -DLLAMA_OPENSSL=ON \
  -DAMDGPU_TARGETS=gfx906 \
  -DCMAKE_BUILD_TYPE=Release \
  -DHIP_COMPILER=clang \
  -DCMAKE_CXX_FLAGS="-O3 -Wno-unused-command-line-argument"
cmake --build build --config Release -j
```

## Running

Recommended environment (each variable enables one of the features above):

```bash
export GGML_ENABLE_CUSTOM_AR=1      # custom multi-GPU AllReduce
export HSA_FORCE_FINE_GRAIN_PCIE=1  # peer-write AllReduce fast path (AMD over PCIe, validated gfx906)
export GPU_MAX_HW_QUEUES=8          # MoE throughput on -tps
export LLAMA_ENABLE_MTP_OPT=1       # MTP optimizations (with --spec-type draft-mtp)
```

On a trimmed ROCm runtime (such as the slim Docker image) also set
`HSA_OVERRIDE_GFX_VERSION=9.0.6` so the runtime recognizes the gfx906 GPU. A full
ROCm install detects it automatically and does not need this.

Always pass `-lm dio` (`--load-mode dio`). mmap on the model file hangs on this
stack. The older `--no-mmap` / `-dio` spellings still parse but are deprecated
upstream, and in `llama-bench` they append two separate load modes, so the old
two-flag form runs every benchmark twice. Select GPUs with `HIP_VISIBLE_DEVICES`
(AMD) or `CUDA_VISIBLE_DEVICES` (NVIDIA); the example commands below use the AMD
form.

```bash
# multi-GPU tensor-parallel server (4 GPUs, full TP)
HIP_VISIBLE_DEVICES=0,1,2,3 llama-server -m model.gguf \
  -ngl 99 -fa 1 -sm tensor -tps 0 -lm dio --host 0.0.0.0 --port 8080

# 8 GPUs as 4 TP groups of 2 (TP=2, PP=4)
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 llama-cli -m model.gguf \
  -ngl 99 -fa 1 -sm tensor -tps 2 -lm dio

# MTP speculative decode (Qwen3.6 dense)
HIP_VISIBLE_DEVICES=0,1 llama-cli -m Qwen3.6-27B-MTP.gguf \
  -ngl 99 -fa 1 -sm tensor --spec-type draft-mtp -lm dio
```
