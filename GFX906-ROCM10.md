# gfx906 + ROCm 10 combo build

This fork ports gfx906 (AMD Instinct MI50/MI60)-specific fixes and optimizations from
several community forks on top of [mxxm-t/mx-llama.cpp](https://github.com/mxxm-t/mx-llama.cpp),
built against a custom ROCm 10 distribution from
[TheRock-gfx906](https://github.com/Wizard815/TheRock-gfx906) (gfx906 isn't an officially
supported target upstream — official support starts at gfx908).

## What's ported here that isn't in mxxm-t/mx-llama.cpp

- `ggml/src/ggml-cuda/rope.cu` — `__sincosf` fusion in `rope_yarn` (from iacopPBK/llama.cpp-gfx906)
- `ggml/src/ggml-cuda/gfx906-sgemm.cuh` — custom tiled SGEMM kernel for gfx906 (from iacopPBK)
- `ggml/src/ggml-cuda/mmq-config-gcn.cuh` — wave64 GCN MMQ tuning table (upstream llama.cpp PR #27841),
  wired as gfx906's fallback instead of rdna2's wave32-tuned table
- `ggml/src/ggml-cuda/mmq-config-gfx906.cuh` — an MI50-measured Q5_K J=64 tile override
  (I=64 vs the GCN table's I=128, +15-35% measured) from mixa3607/ML-gfx906's
  `mxxm-gfx906-kcase.patch`
- `ggml/src/ggml-cuda/gcn_repack/` — Q3_K/Q4_K/Q5_K/Q6_K three-plane weight repack for gfx906
  (from sixvolts/llamacpp-gfx906-furnace), coexisting with mx-llama.cpp's own Q8_0/MXFP4
  two-plane repack; fixed a type-confusion bug where a tensor could be mis-routed to the
  wrong repack scheme's dispatch, and hardened `set_tensor` to fall back to a plain copy
  instead of a hard assert on an unsupported tensor
- `GGML_OP_SOLVE_TRI` crash fix — gfx906's rocBLAS `strsm` fails on large triangular solves
  (affects Gated Delta Network models: Qwen3.5, Qwen3-Next, Kimi Linear); now falls back
  correctly instead of crashing
- Turbo KV cache compression (`--cache-type-k/v turbo2/turbo3/turbo4`) — ~3.3x KV cache
  capacity at a generation-speed cost, via a rotate/quantize/dequantize scheme requiring
  flash attention; ported from moriyasujapan/llamacpp-gfx-906-turbo-gemma4 (a bugfixed
  version of arte-fact/llamacpp-gfx-906-turbo's original). **Multi-GPU layer-split is
  unverified** — the shared rotation-matrix tensor is allocated on whichever GPU's context
  builds first; cross-device access should be handled by ggml's normal scheduler but this
  hasn't been tested past 2 GPUs by any source fork. Test this first on real hardware.
- `-mllvm -amdgpu-sched-strategy=max-ilp` compiler flag recommended at build time (see below)

Flash-attention on gfx906 already uses the native `v_dot2_f32_f16` instruction upstream
(`common.cuh`'s `V_DOT2_F32_F16_AVAILABLE` gate includes `__gfx906__`) — no separate port
needed there.

## Build

Requires a gfx906 ROCm 10 install from
[TheRock-gfx906](https://github.com/Wizard815/TheRock-gfx906) at `/opt/rocm` (or `$ROCM_PATH`).

```bash
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
cmake -S . -B build -GNinja \
    -DGGML_HIP=ON \
    -DGGML_HIP_GRAPHS=ON \
    -DGGML_HIP_RCCL=ON \
    -DAMDGPU_TARGETS=gfx906 \
    -DCMAKE_BUILD_TYPE=Release \
    -DHIP_COMPILER=clang \
    -DCMAKE_CXX_FLAGS="-O3 -Wno-unused-command-line-argument" \
    -DCMAKE_HIP_FLAGS="-mllvm -amdgpu-sched-strategy=max-ilp" \
    -DLLAMA_BUILD_TESTS=OFF
cmake --build build --config Release -j"$(nproc)"
```

A companion Docker setup (multi-stage build against the ROCm 10 tarball, with
`/dev/kfd`/`/dev/dri` passthrough for container-based testing) exists locally but is not
yet published.
