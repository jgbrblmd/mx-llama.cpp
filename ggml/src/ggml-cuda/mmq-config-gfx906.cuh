// gfx906 (Vega20 / MI50, wave64) MMQ config.
//
// Perf-only knobs (nthreads / tile widths) - results are bit-exact vs rdna2,
// which is what upstream routes gfx906 through. Include after mmq-config-rdna2.cuh.
//
// Q8_0: 8 warps (nthreads 512, vs rdna2's 4 = 256), and offer tile widths up to
// J=128 (rdna2 caps its Q8_0 table at 64). The wide tiles are only SELECTED when
// they keep the CUs busy - see the occupancy gate in mul_mat_q_switch_J
// (mmq.cuh), which holds J<=64 for row-sharded (-sm tensor) and MoE shapes and
// lets full-row shapes (1-GPU, -sm layer) take the wider tile.
//
// MXFP4: 8 warps as well - rdna2's table with nthreads overridden, so tile
// widths and layout stay in sync with upstream.
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
    // Q5_K: 8 warps + wide tiles like Q8_0; 3-buffer layout (qs, q8, scales)
    // matches rdna2's, so smem accounting stays consistent. Tune candidate.
    if (type == GGML_TYPE_Q5_K && J >= 8 && J <= 128 && (J % 8) == 0) {
        return ggml_cuda_mmq_config(
            GGML_TYPE_Q5_K, 512, 2, 128, J, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, fallback);
    }
    return ggml_cuda_mmq_get_config_rdna2(type, J, fallback);
}
