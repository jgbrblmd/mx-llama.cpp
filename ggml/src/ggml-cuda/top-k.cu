#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    auto indexes_in = cuda::make_counting_iterator(0);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k,
                         env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxPairs(d_temp_storage, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst,
                         ncols, k, env));
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

#ifndef GGML_CUDA_USE_CUB

// Hierarchical top-k for rows wider than the shared-memory bitonic limit.
// The row is cut into <= 16384-column segments, the existing bitonic sorts
// each segment (staged packed so the row stride matches), the per-segment
// top-k candidates are gathered with global indices, and a final bitonic
// over the n_seg * k candidates selects the row's top-k. Correct as long as
// n_seg * next_pow2(k) fits the shared-memory sort, which the supports_op
// gate guarantees. This keeps the DSV4 lightning-indexer selection
// (ne0 = n_kv/4, crossing 16384 at exactly n_kv 65536) on the GPU - the
// CPU fallback it replaces fractured the graph into per-layer host
// round-trips that broke multi-stage tensor parallel and crawled.

static __global__ void k_topk_gather_candidates(
        const float * x, const int * seg_sorted, float * cand_val, int * cand_idx,
        const int64_t ncols, const int seg_len, const int seg_off, const int kk,
        const int cand_stride, const int cand_off, const int k) {
    const int row = blockIdx.y;
    const int j   = blockIdx.x*blockDim.x + threadIdx.x;
    if (j >= k) {
        return;
    }
    const int64_t crow = (int64_t) row*cand_stride + cand_off;
    if (j >= kk) {
        cand_val[crow + j] = -INFINITY;
        cand_idx[crow + j] = seg_off;
        return;
    }
    const int local = seg_sorted[(int64_t) row*seg_len + j];
    const int gidx  = local + seg_off;
    cand_idx[crow + j] = gidx;
    cand_val[crow + j] = x[(int64_t) row*ncols + gidx];
}

static __global__ void k_topk_remap(
        const int * pos, const int * cand_idx, int * dst,
        const int n_cand, const int k) {
    const int row = blockIdx.y;
    const int j   = blockIdx.x*blockDim.x + threadIdx.x;
    if (j >= k) {
        return;
    }
    dst[(int64_t) row*k + j] = cand_idx[(int64_t) row*n_cand + pos[(int64_t) row*n_cand + j]];
}

static void top_k_hierarchical(ggml_cuda_pool & pool,
                               const float *    src,
                               int *            dst,
                               const int64_t    ncols,
                               const int64_t    nrows,
                               const int64_t    k,
                               cudaStream_t     stream) {
    constexpr int SEGW = 16384;
    const int n_seg  = (int) ((ncols + SEGW - 1) / SEGW);
    const int n_cand = (int) (n_seg * k);

    ggml_cuda_pool_alloc<float> seg_vals_alloc(pool, (int64_t) SEGW * nrows);
    ggml_cuda_pool_alloc<int>   seg_sorted_alloc(pool, (int64_t) SEGW * nrows);
    ggml_cuda_pool_alloc<float> cand_val_alloc(pool, (int64_t) n_cand * nrows);
    ggml_cuda_pool_alloc<int>   cand_idx_alloc(pool, (int64_t) n_cand * nrows);
    ggml_cuda_pool_alloc<int>   pos_alloc(pool, (int64_t) n_cand * nrows);

    const dim3 block(256, 1, 1);
    const dim3 grid((unsigned) ((k + 255) / 256), (unsigned) nrows, 1);

    for (int seg = 0; seg < n_seg; seg++) {
        const int seg_off = seg * SEGW;
        const int seg_len = (int) std::min<int64_t>(SEGW, ncols - seg_off);
        const int kk      = (int) std::min<int64_t>(k, seg_len);

        // pack the segment so the bitonic's row stride matches its ncols
        CUDA_CHECK(cudaMemcpy2DAsync(seg_vals_alloc.get(), seg_len * sizeof(float),
                                     src + seg_off, ncols * sizeof(float),
                                     seg_len * sizeof(float), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
        argsort_f32_i32_cuda_bitonic(seg_vals_alloc.get(), seg_sorted_alloc.get(),
                                     seg_len, nrows, GGML_SORT_ORDER_DESC, stream);
        k_topk_gather_candidates<<<grid, block, 0, stream>>>(
            src, seg_sorted_alloc.get(), cand_val_alloc.get(), cand_idx_alloc.get(),
            ncols, seg_len, seg_off, kk, n_cand, seg * (int) k, (int) k);
    }

    argsort_f32_i32_cuda_bitonic(cand_val_alloc.get(), pos_alloc.get(),
                                 n_cand, nrows, GGML_SORT_ORDER_DESC, stream);
    k_topk_remap<<<grid, block, 0, stream>>>(
        pos_alloc.get(), cand_idx_alloc.get(), dst, n_cand, (int) k);
}

#endif // !GGML_CUDA_USE_CUB

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
#ifdef CUB_TOP_K_AVAILABLE
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    // TODO: investigate if there exists a point where parallelized argsort is faster than sequential top-k
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
    if (ncols <= 16384) {
        ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
        int *                     tmp_dst = temp_dst_alloc.get();
        argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                     cudaMemcpyDeviceToDevice, stream));
    } else {
        top_k_hierarchical(pool, src0_d, dst_d, ncols, nrows, k, stream);
    }
#endif
}
