// Dense (non-MoE) entry point for repacked Q8_0 weights: quantize src1, then dispatch
// the mat-vec (single token) or tiled GEMM (multi-token) path per 2D slice.
#include "repack.cuh"
#include "repack-common.cuh"
#include "repack-kernels.cuh"
#include "../quantize.cuh"
#include "../mmq.cuh"

static void ggml_cuda_mul_mat_repacked_slice(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const uint8_t * w, const block_q8_1 * xq,
        float * dst_d, int64_t ne00, int64_t ne01, int64_t ne11,
        cudaStream_t stream);

void ggml_cuda_mul_mat_repacked(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(dst->nb[1]  == (size_t) dst->ne[0] * sizeof(float));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];
    GGML_ASSERT(ne10 == ne00);

    cudaStream_t stream = ctx.stream();

    // Views are re-packed into a persistent cache buffer (pool allocs are invalid
    // inside CUDA graph capture).
    const uint8_t * w;
    if (src0->view_src != nullptr && ggml_cuda_repack_tensor_supported(src0->view_src)) {
        w = repack_q8_0_view_get_cached(src0, src0->view_src, stream);
    } else {
        w = (const uint8_t *) src0->data;
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const int64_t x_stride    = ne10_padded / QK8_1;

    if (ne11 == 1) {
        // Single token: plain Q8_1, mat-vec per slice.
        ggml_cuda_pool_alloc<block_q8_1> src1_q8_1(ctx.pool(),
            ne13 * ne12 * x_stride);
        {
            const int64_t s11 = src1->nb[1] / sizeof(float);
            const int64_t s12 = src1->nb[2] / sizeof(float);
            const int64_t s13 = src1->nb[3] / sizeof(float);
            quantize_row_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1.get(),
                src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
        }
        for (int64_t i3 = 0; i3 < ne13; i3++) {
        for (int64_t i2 = 0; i2 < ne12; i2++) {
            const block_q8_1 * xq = src1_q8_1.get()
                              + (i3 * ne12 + i2) * x_stride;
            float * dst_d = (float *)((char *) dst->data + i3 * dst->nb[3] + i2 * dst->nb[2]);
            ggml_cuda_mul_mat_repacked_slice(ctx, src0, w, xq, dst_d,
                ne00, ne01, ne11, stream);
        }
        }
        return;
    }

    // Multi-token: grouped MMQ Q8_1 layout for the tiled GEMM.
    const uint64_t n_groups   = (uint64_t) ne10_padded / (4 * QK8_1);

    ggml_cuda_pool_alloc<block_q8_1_mmq_h> src1_q8_1(ctx.pool(),
        ne13 * ne12 * ne11 * n_groups);
    {
        const int64_t s11 = src1->nb[1] / sizeof(float);
        const int64_t s12 = src1->nb[2] / sizeof(float);
        const int64_t s13 = src1->nb[3] / sizeof(float);
        quantize_mmq_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1.get(),
            src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    }
    for (int64_t i3 = 0; i3 < ne13; i3++) {
    for (int64_t i2 = 0; i2 < ne12; i2++) {
        const block_q8_1_mmq_h * slice_base = src1_q8_1.get()
                              + (i3 * ne12 + i2) * ne11 * n_groups;
        const block_q8_1 * xq = reinterpret_cast<const block_q8_1 *>(slice_base);
        float * dst_d = (float *)((char *) dst->data + i3 * dst->nb[3] + i2 * dst->nb[2]);
        ggml_cuda_mul_mat_repacked_slice(ctx, src0, w, xq, dst_d,
            ne00, ne01, ne11, stream);
    }
    }
}

// Dispatch one 2D slice: mat-vec vs tiled GEMM.
static void ggml_cuda_mul_mat_repacked_slice(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const uint8_t * w, const block_q8_1 * xq,
        float * dst_d, const int64_t ne00, const int64_t ne01, const int64_t ne11,
        cudaStream_t stream) {
    if (ne11 == 1) {
        switch (src0->type) {
            case GGML_TYPE_Q8_0: {
                const dim3 grid((ne01 + 15) / 16, 1, 1);
                mul_mat_vec_q8_0_repacked<16, 16, false><<<grid, 1024, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    nullptr, nullptr, nullptr, 0, 1, 0, 0, 0,
                    nullptr, nullptr, nullptr, GGML_GLU_OP_REGLU);
            } break;
            default: GGML_ABORT("unsupported repack type");
        }
        return;
    }

    // Multi-token: 64-wide GEMM for large batches, 32-wide for small ones.
    const int nrl  = MMQ_RP_Q8_NROW_LANES;
    const int mmq_bm = MMQ_RP_Q8_BM;
    if (ne11 >= 128) {
        const int bn = 64 * MMQ_RP_Q8_TN;
        const dim3 grid((ne01 + mmq_bm - 1) / mmq_bm,
                        (ne11 + bn - 1) / bn, 1);
        mmq_gemm_q8_0_repacked<false, MMQ_RP_Q8_TN, nrl><<<grid, dim3(64, nrl), 0, stream>>>(
            w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
            nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
    } else {
        const int bn = 32;
        const dim3 grid((ne01 + mmq_bm - 1) / mmq_bm,
                        (ne11 + bn - 1) / bn, 1);
        mmq_gemm_q8_0_repacked_w32<false, 1, nrl*2><<<grid, dim3(32, nrl*2), 0, stream>>>(
            w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) ne11,
            nullptr, nullptr, nullptr, nullptr, nullptr, 0, 0, (uint32_t) ne01);
    }
    GGML_UNUSED(ctx);
}

// Fused MMV entry for repacked weights (dense): single-token only, fuses the up and
// gate dot products (shared quantized input) plus optional biases and the GLU op.
// Dispatch: ggml_cuda_try_fuse `{op,op,GLU}` and `{op,ADD}` repack branches.
void ggml_cuda_mul_mat_vec_repacked_fused(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
        const ggml_cuda_mm_fusion_args_host * fusion) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(dst->nb[1]  == (size_t) dst->ne[0] * sizeof(float));
    GGML_ASSERT(dst->ne[1] == 1);   // fused MMV is single-token only
    GGML_ASSERT(fusion != nullptr);

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne10 = src1->ne[0];
    const int64_t ne12 = src1->ne[2];
    const int64_t ne13 = src1->ne[3];
    GGML_ASSERT(ne10 == ne00);

    cudaStream_t stream = ctx.stream();

    const uint8_t * w;
    if (src0->view_src != nullptr && ggml_cuda_repack_tensor_supported(src0->view_src)) {
        w = repack_q8_0_view_get_cached(src0, src0->view_src, stream);
    } else {
        w = (const uint8_t *) src0->data;
    }

    const uint8_t * w_gate = nullptr;
    if (fusion->gate != nullptr) {
        const ggml_tensor * gate = fusion->gate;
        if (gate->view_src != nullptr && ggml_cuda_repack_tensor_supported(gate->view_src)) {
            w_gate = repack_q8_0_view_get_cached(gate, gate->view_src, stream);
        } else {
            w_gate = (const uint8_t *) gate->data;
        }
    }

    const float * x_bias     = fusion->x_bias     ? (const float *) fusion->x_bias->data     : nullptr;
    const float * gate_bias  = fusion->gate_bias  ? (const float *) fusion->gate_bias->data  : nullptr;
    const ggml_glu_op glu_op = fusion->glu_op;

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const int64_t x_stride    = ne10_padded / QK8_1;

    ggml_cuda_pool_alloc<block_q8_1> src1_q8_1(ctx.pool(),
        ne13 * ne12 * x_stride);
    {
        const int64_t s11 = src1->nb[1] / sizeof(float);
        const int64_t s12 = src1->nb[2] / sizeof(float);
        const int64_t s13 = src1->nb[3] / sizeof(float);
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1.get(),
            src0->type, ne10, s11, s12, s13, ne10_padded, 1, ne12, ne13, stream);
    }
    for (int64_t i3 = 0; i3 < ne13; i3++) {
    for (int64_t i2 = 0; i2 < ne12; i2++) {
        const block_q8_1 * xq = src1_q8_1.get()
                          + (i3 * ne12 + i2) * x_stride;
        float * dst_d = (float *)((char *) dst->data + i3 * dst->nb[3] + i2 * dst->nb[2]);
        // Bias is shaped like the single-token dst ([ne0, 1, ne2]): per-channel stride ne01.
        const float * x_bias_s    = x_bias    ? x_bias    + (i3 * ne12 + i2) * ne01 : nullptr;
        const float * gate_bias_s = gate_bias ? gate_bias + (i3 * ne12 + i2) * ne01 : nullptr;

        const dim3 grid((ne01 + 15) / 16, 1, 1);
        mul_mat_vec_q8_0_repacked<16, 16, false, 64, true><<<grid, 1024, 0, stream>>>(
            w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
            nullptr, nullptr, nullptr, 0, 1, 0, 0, 0,
            w_gate, x_bias_s, gate_bias_s, glu_op);
    }
    }
}
