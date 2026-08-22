// Host-side support for the Q8_0 repacked-weight path: tensor support check, host
// repack routine, and the persistent per-view cache shared by dense and MoE paths.
#include "repack.cuh"
#include "repack-common.cuh"

#include <cstring>
#include <map>
#include <mutex>

bool ggml_cuda_repack_tensor_supported(const ggml_tensor * t) {
    // Views are never in repacked layout (the scale-plane offset needs the FULL ne1).
    if (t->view_src != nullptr) return false;
    if ((ggml_n_dims(t) != 2 && ggml_n_dims(t) != 3) || !ggml_is_contiguous(t)) {
        return false;
    }
    switch (t->type) {
        case GGML_TYPE_Q8_0: {
            return t->ne[0] % 32 == 0;
        }
        default:             return false;
    }
}

bool ggml_cuda_repack_mmv_fusion_width_ok(int64_t n_tokens, bool has_ids) {
    // MoE fuses across the whole narrow range: its grid is one block per
    // assignment, so at these widths launch overhead dominates and folding
    // three launches into one pays. Dense does NOT: its grid already spans
    // the full row count, there is ample parallelism to hide latency, and
    // halving the block count to double the work per block costs more than
    // the saved launches return. Measured on Qwen3.6-27B-MTP-Q8_0, pp512 at
    // that ubatch, fused against unfused: +4.6% at 2 tokens, -2.7% at 3,
    // +2.4% at 4 - and a speculative verify step is exactly 3 wide at the
    // default draft depth, so dense fusion loses where it would be used.
    const int64_t max_tokens = has_ids ? MMQ_RP_Q8_MOE_MMV_MAX_TOKENS : 1;
    return n_tokens >= 1 && n_tokens <= max_tokens;
}

bool ggml_cuda_repack_mul_mat_should_fire(const ggml_tensor * src0) {
    if (src0->buffer == nullptr || !ggml_backend_buft_is_cuda_repack(ggml_backend_buffer_get_type(src0->buffer))) {
        return false;
    }
    if (ggml_cuda_repack_tensor_supported(src0)) {
        return true;              // full tensor in repacked layout
    }
    return src0->view_src != nullptr && ggml_cuda_repack_tensor_supported(src0->view_src);
}

// Host repack of one Q8_0 matrix: qs plane [ne1 x nsp x 32] then f16 scale plane
// [ne1 x nsp]; padding sub-blocks are left zeroed.
void repack_q8_0_host(const block_q8_0 * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    // a non-multiple ne0 would silently drop the row tail below, so fail loudly instead
    GGML_ASSERT(ne0 % 32 == 0);
    const int64_t n_blocks = ne0 / 32;
    const int64_t qs_str   = repack_qs_stride(ne0);
    const size_t  qs_len   = (size_t) ne1 * qs_str;

    memset(dst, 0, qs_len + (size_t) ne1 * n_blocks * 2);

    for (int64_t row = 0; row < ne1; row++) {
        for (int64_t blk = 0; blk < n_blocks; blk++) {
            const block_q8_0 * b = &blocks[row * n_blocks + blk];
            memcpy(dst + (size_t) row * qs_str + (size_t) blk * 32, b->qs, 32);
            memcpy(dst + qs_len + (size_t)(row * n_blocks + blk) * 2, &b->d, 2);
        }
    }
}

// Persistent cache for re-packed view buffers. Pool allocs are invalid inside CUDA
// graph capture, so buffers are cudaMalloc'd once per unique view and never freed.
struct RepackViewCacheKey {
    const void * view_data;
    int64_t ne0;
    int64_t ne1;
    int64_t ne2;
    bool operator<(const RepackViewCacheKey & o) const {
        if (view_data != o.view_data) return view_data < o.view_data;
        if (ne0 != o.ne0) return ne0 < o.ne0;
        if (ne1 != o.ne1) return ne1 < o.ne1;
        return ne2 < o.ne2;
    }
};

struct RepackViewCacheEntry {
    uint8_t * d_ptr;
    size_t    size;
};

static std::map<RepackViewCacheKey, RepackViewCacheEntry> s_view_cache;
static std::mutex s_view_cache_mutex;

// Get or create a persistent device buffer for a re-packed view (source may be a
// graph-captured stream; src/dst addresses are stable because the entry persists).
const uint8_t * repack_q8_0_view_get_cached(
        const ggml_tensor * view, const ggml_tensor * base,
        cudaStream_t stream) {
    const int64_t ne0_v = view->ne[0];
    const int64_t ne1_v = view->ne[1];
    const int64_t ne2_v = view->ne[2];
    const int64_t ne0_b = base->ne[0];
    const int64_t ne1_b = base->ne[1];
    const int64_t ne2_b = base->ne[2];
    GGML_ASSERT(ne0_v == ne0_b);
    GGML_ASSERT(base->view_src == nullptr);

    const int64_t qs_str  = repack_qs_stride(ne0_v);
    const int64_t n_sub   = ne0_v / 32;
    const uint8_t * base_ptr = (const uint8_t *) base->data;
    const uint8_t * view_ptr = (const uint8_t *) view->data;

    const int64_t row_start = (int64_t)(view_ptr - base_ptr) / qs_str;
    GGML_ASSERT((view_ptr - base_ptr) % qs_str == 0);

    // Copy every expert in the view (ne1_v x ne2_v), not just ne1_v rows.
    const size_t qs_size = (size_t) ne1_v * ne2_v * qs_str;
    const size_t sc_size = (size_t) ne1_v * ne2_v * n_sub * 2;
    const size_t total   = qs_size + sc_size;

    RepackViewCacheKey key{ view->data, ne0_v, ne1_v, ne2_v };

    std::lock_guard<std::mutex> lock(s_view_cache_mutex);
    auto it = s_view_cache.find(key);
    if (it != s_view_cache.end()) {
        return it->second.d_ptr;
    }

    uint8_t * d_ptr;
    CUDA_CHECK(cudaMalloc(&d_ptr, total));

    // QS plane: copy from the view's position in the base QS plane.
    CUDA_CHECK(cudaMemcpyAsync(d_ptr, view_ptr, qs_size,
        cudaMemcpyDeviceToDevice, stream));

    // Scale plane: starts after the FULL base QS plane.
    const uint8_t * src_scales = base_ptr + (size_t) ne1_b * ne2_b * qs_str
                                       + (size_t) row_start * n_sub * 2;
    CUDA_CHECK(cudaMemcpyAsync(d_ptr + qs_size, src_scales, sc_size,
        cudaMemcpyDeviceToDevice, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    s_view_cache[key] = { d_ptr, total };
    return d_ptr;
}

void repack_view_cache_purge(int device, const void * base, size_t size) {
    if (base == nullptr || size == 0) {
        return;
    }
    const uint8_t * lo = (const uint8_t *) base;
    const uint8_t * hi = lo + size;
    ggml_cuda_set_device(device);
    std::lock_guard<std::mutex> lock(s_view_cache_mutex);
    for (auto it = s_view_cache.begin(); it != s_view_cache.end(); ) {
        const uint8_t * v = (const uint8_t *) it->first.view_data;
        if (v >= lo && v < hi) {
            CUDA_CHECK(cudaFree(it->second.d_ptr));
            it = s_view_cache.erase(it);
        } else {
            ++it;
        }
    }
}
