// gfx906 (Vega20 / MI50, wave64) MMQ config.
//
// Perf-only knobs (nthreads / tile widths). Q8_0/MXFP4 below keep their own
// hand-tuned overrides; everything else (Q2_K..Q6_K, Q4_0/1, Q5_0/1, IQ-types,
// NVFP4 - notably Q4_K, the common quant for this hardware) falls through to
// mmq-config-gcn.cuh's dedicated wave64 table (ported from upstream PR #27841)
// instead of rdna2's wave32-tuned one. Include after mmq-config-gcn.cuh.
//
// Q8_0: 8 warps (nthreads 512, vs rdna2's 4 = 256), and offer tile widths up to
// J=128 (rdna2 caps its Q8_0 table at 64). The wide tiles are only SELECTED when
// they keep the CUs busy - see the occupancy gate in mul_mat_q_switch_J
// (mmq.cuh), which holds J<=64 for row-sharded (-sm tensor) and MoE shapes and
// lets full-row shapes (1-GPU, -sm layer) take the wider tile.
//
// MXFP4: 8 warps as well - rdna2's table with nthreads overridden, so tile
// widths and layout stay in sync with upstream.
//
// Q5_K, J=64: mmq-config-gcn.cuh's own entry uses I=128 here; MI50 measurement
// (mixa3607/ML-gfx906, mxxm-gfx906-kcase.patch) found I=64 ~15-35% faster at
// this tile width - same accumulator-array/LDS-tile halving that already made
// Q6_K's own J=64 entry I=64 in the GCN table. Q4_K and Q6_K need no override:
// their GCN-table entries already match the measured-best I for every J.
static constexpr __host__ __device__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config_gfx906(ggml_type type, int J, bool fallback) {
    if (type == GGML_TYPE_Q8_0 && J >= 8 && J <= 128 && (J % 8) == 0) {
        return ggml_cuda_mmq_config(
            GGML_TYPE_Q8_0, 512, 2, 128, J, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, fallback);
    }
    if (type == GGML_TYPE_MXFP4) {
        const ggml_cuda_mmq_config rdna2 = ggml_cuda_mmq_get_config_rdna2(type, J, fallback);
        if (rdna2.type == GGML_TYPE_COUNT) {
            return rdna2;
        }
        return ggml_cuda_mmq_config(
            rdna2.type, 512, rdna2.occupancy, rdna2.I, rdna2.J, rdna2.sram_layout, rdna2.K_vram, rdna2.stream_k, rdna2.fallback);
    }
    // NVFP4: 8 warps as well. rdna2 caps its NVFP4 table at J=64; extend to
    // J=128 like Q8_0 - the occupancy gate in mul_mat_q_switch_J (mmq.cuh)
    // keeps wide tiles for shapes that fill the CUs, so results stay identical.
    if (type == GGML_TYPE_NVFP4 && J >= 8 && J <= 128 && (J % 8) == 0) {
        return ggml_cuda_mmq_config(
            GGML_TYPE_NVFP4, 512, 2, 128, J, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, false, fallback);
    }
    if (type == GGML_TYPE_NVFP4_E8M0 && J >= 8 && J <= 128 && (J % 8) == 0) {
        return ggml_cuda_mmq_config(
            GGML_TYPE_NVFP4_E8M0, 512, 2, 128, J, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, false, fallback);
    }
    // FP8: same layout as Q8_0 (int8 data, Q8_1 layout)
    if (type == GGML_TYPE_Q8_0_ROCMFPX && J >= 8 && J <= 128 && (J % 8) == 0) {
        return ggml_cuda_mmq_config(
            GGML_TYPE_Q8_0_ROCMFPX, 512, 2, 128, J, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, fallback);
    }
    // FP2: 16-wide dp4a path, same I/nthreads as Q8_0 on gfx906
    if ((type == GGML_TYPE_Q2_0_ROCMFPX || type == GGML_TYPE_Q2_0_ROCMFPX_AFFINE) && J >= 8 && J <= 128 && (J % 8) == 0) {
        return ggml_cuda_mmq_config(
            type, 512, 2, 128, J, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, fallback);
    }
    // FP3: 16-wide dp4a path, same I/nthreads as Q8_0 on gfx906
    if (type == GGML_TYPE_Q3_0_ROCMFPX && J >= 8 && J <= 128 && (J % 8) == 0) {
        return ggml_cuda_mmq_config(
            GGML_TYPE_Q3_0_ROCMFPX, 512, 2, 128, J, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, fallback);
    }
    // FP6: 16-wide dp4a path, same I/nthreads as Q8_0 on gfx906
    if (type == GGML_TYPE_Q6_0_ROCMFPX && J >= 8 && J <= 128 && (J % 8) == 0) {
        return ggml_cuda_mmq_config(
            GGML_TYPE_Q6_0_ROCMFPX, 512, 2, 128, J, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, fallback);
    }
    // ROCmFP4 standard: 16-wide dp4a path, same nthreads as Q8_0 on gfx906
    if (type == GGML_TYPE_Q4_0_ROCMFP4 && J >= 8 && J <= 128 && (J % 8) == 0) {
        return ggml_cuda_mmq_config(
            GGML_TYPE_Q4_0_ROCMFP4, 512, 2, 128, J, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, fallback);
    }
    // ROCmFP4 fast: same as Q8_0 (single scale, int8-like dp4a)
    if (type == GGML_TYPE_Q4_0_ROCMFP4_FAST && J >= 8 && J <= 128 && (J % 8) == 0) {
        return ggml_cuda_mmq_config(
            GGML_TYPE_Q4_0_ROCMFP4_FAST, 512, 2, 128, J, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, fallback);
    }
    // Q5_K: J=64 measured faster with I=64/256 threads on MI50 (mxxm-gfx906-kcase.patch);
    // everything else falls through to the dedicated wave64 GCN table.
    CASE(GGML_TYPE_Q5_K, 256, 2, 64, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, true);
    CASE(GGML_TYPE_Q5_K, 256, 2, 64, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false);
    return ggml_cuda_mmq_get_config_gcn(type, J, fallback);
}
