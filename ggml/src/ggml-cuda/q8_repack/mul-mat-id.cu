// MoE entry point (mul_mat_id) for repacked Q8_0 weights: route tokens to their
// experts, then run the mat-vec (single token) or tiled GEMM (multi-token) path.
#include "repack.cuh"
#include "repack-common.cuh"
#include "repack-kernels.cuh"
#include "../quantize.cuh"
#include "../mmq.cuh"
#include "../mmid.cuh"

void ggml_cuda_mul_mat_id_repacked(ggml_backend_cuda_context & ctx,
        const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * ids,
        ggml_tensor * dst) {
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(ids->type  == GGML_TYPE_I32);
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(ids->nb[0]  == sizeof(int32_t));
    GGML_ASSERT(src1->ne[3] == 1 && dst->ne[3] == 1);
    GGML_ASSERT(src1->nb[2] == src1->nb[1] * src1->ne[1]);
    GGML_ASSERT(dst->nb[2]  == dst->nb[1]  * dst->ne[1]);
    GGML_ASSERT(dst->nb[1]  == (size_t) dst->ne[0] * sizeof(float));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    const int64_t ne10 = src1->ne[0];
    GGML_ASSERT(ne10 == ne00);
    const int64_t n_expert_used = ids->ne[0];
    const int64_t n_tokens      = ids->ne[1];
    const int64_t n_assign      = n_expert_used * n_tokens;

    cudaStream_t stream = ctx.stream();

    const uint8_t * w;
    if (src0->view_src != nullptr && ggml_cuda_repack_tensor_supported(src0->view_src)) {
        w = repack_q8_0_view_get_cached(src0, src0->view_src, stream);
    } else {
        w = (const uint8_t *) src0->data;
    }
    float * dst_d = (float *) dst->data;
    const size_t expert_stride = repack_gcn_nbytes(src0->type, ne00, ne01);
    const uint32_t dst_s1 = dst->nb[1] / sizeof(float);

    ggml_cuda_pool_alloc<int32_t> ids_src1(ctx.pool(), n_assign);
    ggml_cuda_pool_alloc<int32_t> ids_dst (ctx.pool(), n_assign);
    ggml_cuda_pool_alloc<int32_t> expert_bounds(ctx.pool(), ne02 + 1);
    if (n_tokens > 1) {
        const int si1  = ids->nb[1] / sizeof(int32_t);
        const int sis1 = src1->nb[2] / src1->nb[1];
        ggml_cuda_launch_mm_ids_helper((const int32_t *) ids->data,
            ids_src1.get(), ids_dst.get(), expert_bounds.get(),
            ne02, n_tokens, n_expert_used, src1->ne[1], si1, sis1, false, stream);
        CUDA_CHECK(cudaGetLastError());
    }

    const int64_t ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING);
    const int64_t x_stride    = ne10_padded / QK8_1;
    const int64_t n_cols      = src1->ne[1] * src1->ne[2];

    const int64_t s11 = src1->nb[1] / sizeof(float);

    if (n_tokens == 1) {
        ggml_cuda_pool_alloc<block_q8_1> src1_q8_1(ctx.pool(),
            (size_t) n_cols * x_stride);
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1.get(),
            src0->type, ne10, s11, s11 * n_cols, s11 * n_cols, ne10_padded,
            n_cols, 1, 1, stream);
        const block_q8_1 * xq = src1_q8_1.get();
        const uint32_t nchannels_y = (uint32_t) src1->ne[1];
        const uint32_t xs_id       = (uint32_t) x_stride;
        switch (src0->type) {
            case GGML_TYPE_Q8_0: {
                const dim3 grid((ne01 + 63) / 64, n_assign, 1);
                mul_mat_vec_q8_0_repacked<64, 16, true, 16><<<grid, 1024, 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01,
                    (const int32_t *) ids->data, nullptr, nullptr,
                    (uint32_t) ne02, nchannels_y, expert_stride, xs_id, dst_s1,
                    nullptr, nullptr, nullptr, GGML_GLU_OP_REGLU);
            } break;
            default: GGML_ABORT("unsupported repack type");
        }
        return;
    }

    const uint64_t n_groups   = (uint64_t) ne10_padded / (4 * QK8_1);

    ggml_cuda_pool_alloc<char> src1_q8_1(ctx.pool(),
        (size_t) n_cols * n_groups * sizeof(block_q8_1_mmq_h));
    {
        quantize_mmq_q8_1_cuda((const float *) src1->data, nullptr, src1_q8_1.get(),
            src0->type, ne10, s11, s11 * n_cols, s11 * n_cols, ne10_padded,
            n_cols, 1, 1, stream);
    }
    const block_q8_1 * xq = (const block_q8_1 *) src1_q8_1.get();

    // Small ubatch: 32-wide tile GEMM pins column blocks tightly (BN = 32*TN) and
    // reaches occupancy 3; large ubatch prefers the 64-wide tile (BN = 64*TN).
    // The tile token width is baked into repack_tile_off<BN>, so each path gets its
    // own prefix-sum + tile-meta buffers.
    constexpr int BN_ID   = 64 * MMQ_RP_Q8_TN;      // 64-wide tile
    constexpr int BN_W32  = 32 * 1;                 // 32-wide tile, TN==1
    const bool use_w32 = n_tokens < MMQ_RP_Q8_MOE_W32_MAX_TOKENS;

    switch (src0->type) {
        case GGML_TYPE_Q8_0: {
            if (use_w32) {
                const int64_t max_tiles_w32 = n_assign / BN_W32 + ne02;
                ggml_cuda_pool_alloc<int32_t>          tile_off_w32 (ctx.pool(), ne02 + 1);
                ggml_cuda_pool_alloc<repack_tile_meta> tile_meta_w32(ctx.pool(), max_tiles_w32);
                repack_tile_off<BN_W32><<<1, 1, 0, stream>>>(expert_bounds.get(), tile_off_w32.get(), tile_meta_w32.get(), ne02);
                const dim3 grid((ne01 + MMQ_RP_Q8_BM - 1) / MMQ_RP_Q8_BM, max_tiles_w32, 1);
                mmq_gemm_q8_0_repacked_w32<true, 1, MMQ_RP_Q8_NROW_LANES * 2><<<grid, dim3(32, MMQ_RP_Q8_NROW_LANES * 2), 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) n_cols,
                    ids_src1.get(), ids_dst.get(), expert_bounds.get(), tile_off_w32.get(), tile_meta_w32.get(),
                    (uint32_t) ne02, expert_stride, dst_s1);
            } else {
                const int64_t max_tiles = n_assign / BN_ID + ne02;
                ggml_cuda_pool_alloc<int32_t>           tile_off (ctx.pool(), ne02 + 1);
                ggml_cuda_pool_alloc<repack_tile_meta>  tile_meta(ctx.pool(), max_tiles);
                repack_tile_off<BN_ID><<<1, 1, 0, stream>>>(expert_bounds.get(), tile_off.get(), tile_meta.get(), ne02);
                // Loose grid.y bound; the kernel's blockIdx.y >= tile_off[n_expert] guard skips slack.
                const dim3 grid((ne01 + MMQ_RP_Q8_BM - 1) / MMQ_RP_Q8_BM, max_tiles, 1);
                mmq_gemm_q8_0_repacked<true, MMQ_RP_Q8_TN, MMQ_RP_Q8_NROW_LANES><<<grid, dim3(64, MMQ_RP_Q8_NROW_LANES), 0, stream>>>(
                    w, xq, dst_d, (uint32_t) ne00, (uint32_t) ne01, (uint32_t) n_cols,
                    ids_src1.get(), ids_dst.get(), expert_bounds.get(), tile_off.get(), tile_meta.get(),
                    (uint32_t) ne02, expert_stride, dst_s1);
            }
        } break;
        default: GGML_ABORT("unsupported repack type");
    }
}
