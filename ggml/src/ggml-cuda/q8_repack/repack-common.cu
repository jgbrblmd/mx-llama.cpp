// Host-side support for the Q8_0 repacked-weight path: tensor support check, host
// repack routine, and the persistent per-view cache shared by dense and MoE paths.
#include "repack.cuh"
#include "repack-common.cuh"

#include <cstdlib>

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
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_IQ4_NL: {
            return t->ne[0] % 32 == 0;
        }
        case GGML_TYPE_CT_INT4: {
            // whole 128-weight blocks: the repack maps each block to 4 Q8_0 sub-blocks,
            // a partial trailing block would repack to garbage
            return t->ne[0] % 128 == 0;
        }
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K: {
            // whole super-blocks, and no views: the multi-plane layouts have no view splice yet
            return t->ne[0] % 256 == 0 && t->view_src == nullptr;
        }
        case GGML_TYPE_Q5_1: {
            // 32-value blocks, multi-plane like the K-quants, so no view splice either
            return t->ne[0] % 32 == 0 && t->view_src == nullptr;
        }
        default:             return false;
    }
}

bool ggml_cuda_repack_mmv_fusion_width_ok(int64_t n_tokens, bool has_ids, ggml_type wt) {
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
    GGML_UNUSED(wt);
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

// The FUSED mat-vec path routes each type to its own kernel: Q8_0 keeps the tuned
// bespoke mul_mat_vec_q8_0_repacked, every other repack type takes the generic
// mul_mat_vec_rp (which reads through rp_traits, so the nibble-plane types and the
// affine CT_INT4 are covered). Offered to the dense FFN up/gate/GLU sites.
bool ggml_cuda_repack_mmv_fusion_supported(const ggml_tensor * src0) {
    const ggml_tensor * t = src0->view_src != nullptr ? src0->view_src : src0;
    return (t->type == GGML_TYPE_Q8_0 || t->type == GGML_TYPE_MXFP4 || t->type == GGML_TYPE_CT_INT4) &&
           ggml_cuda_repack_mul_mat_should_fire(src0);
}

// The MoE up/gate fusion has a fused mat-vec for every repacked type, so the expert path admits all of them.
// The dense fusion stays on the predicate above until its K-quant and bias variants have their own gate.
// GGML_CUDA_REPACK_KQUANT_MOE_FUSION=0 keeps the expert path on the Q8_0 and MXFP4 admission.
bool ggml_cuda_repack_mmv_id_fusion_supported(const ggml_tensor * src0) {
    static const bool kquant = [] {
        const char * env = getenv("GGML_CUDA_REPACK_KQUANT_MOE_FUSION");
        return env == nullptr || atoi(env) != 0;
    }();
    const ggml_tensor * t = src0->view_src != nullptr ? src0->view_src : src0;
    const bool base = t->type == GGML_TYPE_Q8_0 || t->type == GGML_TYPE_MXFP4;
    // Each of these four types has been gated on a real MoE model: deterministic, unfused path unchanged, width-1 PPL within the repack's own spread.
    const bool kq   = t->type == GGML_TYPE_Q4_K || t->type == GGML_TYPE_Q5_K || t->type == GGML_TYPE_Q6_K || t->type == GGML_TYPE_IQ4_NL;
    return (base || (kquant && kq)) && ggml_cuda_repack_mul_mat_should_fire(src0);
}

// Host repack of one CT_INT4 matrix: dequantize each 128-weight block (1 x f16
// scale + 16 x int32, LSB-first nibbles) into 4 Q8_0 sub-blocks and store them
// in the Q8_0 two-plane layout.
// Host repack of one CT_INT4 matrix: de-aliased nibble rows [ne1 x rs, byte i =
// code i | code (16+i) << 4, raw codes 0..15] then a 2-byte f16 scale plane [ne1 x
// ne0/128] (one entry per 128-weight group, not replicated over the 4 sub-blocks).
// The raw codes ride the affine epilogue (dequant (c-8)*d) on the kernel side.
void repack_ct_int4_host(const block_ct_int4 * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    GGML_ASSERT(ne0 % 32 == 0);
    const int64_t n_blocks = ne0 / 32;
    const int64_t n_ct     = ne0 / 128;
    const int64_t rs       = repack_qs_stride(ne0 / 2);   // nibble row stride (bytes)
    const size_t  qs_len   = (size_t) ne1 * rs;

    memset(dst, 0, qs_len + (size_t) ne1 * n_ct * 2);

    for (int64_t row = 0; row < ne1; row++) {
        for (int64_t blk = 0; blk < n_ct; blk++) {
            const block_ct_int4 & c = blocks[row * n_ct + blk];
            const uint16_t d_bits = *reinterpret_cast<const uint16_t *> (&c.d);
            // 4 sub-blocks of 32; sub s holds the 128-block positions s*32 + k.
            for (int s = 0; s < 4; s++) {
                for (int i = 0; i < 16; i++) {
                    const int p0 = s * 32 + i;
                    const int p1 = p0 + 16;
                    const uint32_t w0 = c.qs[p0 / 8];
                    const uint32_t w1 = c.qs[p1 / 8];
                    const uint32_t lo = (w0 >> ((p0 % 8) * 4)) & 0x0F;
                    const uint32_t hi = (w1 >> ((p1 % 8) * 4)) & 0x0F;
                    dst[(size_t) row * rs + (size_t) (blk * 4 + s) * 16 + i] = (uint8_t) (lo | (hi << 4));
                }
            }
            dst[qs_len + (size_t) (row * n_ct + blk) * 2]     = (uint8_t) d_bits;
            dst[qs_len + (size_t) (row * n_ct + blk) * 2 + 1] = (uint8_t) (d_bits >> 8);
        }
    }
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

// Host repack of one MXFP4 matrix: nibble rows [ne1 x qs_str, de-aliased like
// Q8_0's qs rows] then a 1-byte e8m0 plane [ne1 x ne0/32]. The nibbles are
// copied VERBATIM - the low/high split is undone on the device by
// rp_mxfp4_expand, so the packed bytes stay a single aligned uint4 load.
// Padding bytes are left zeroed. NOTE: a zero e8m0 byte decodes to 2^-127, not
// 0, so device code must never dot padding nibbles against live data - the
// kernels guard with sb < n_sub, not with the scale.
void repack_mxfp4_host(const block_mxfp4 * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    GGML_ASSERT(ne0 % 32 == 0);
    const int64_t n_blocks = ne0 / 32;
    const int64_t qs_str   = repack_qs_row_stride(GGML_TYPE_MXFP4, ne0);
    const size_t  qs_len   = (size_t) ne1 * qs_str;

    memset(dst, 0, qs_len + (size_t) ne1 * n_blocks);

    for (int64_t row = 0; row < ne1; row++) {
        for (int64_t blk = 0; blk < n_blocks; blk++) {
            const block_mxfp4 * b = &blocks[row * n_blocks + blk];
            memcpy(dst + (size_t) row * qs_str + (size_t) blk * 16, b->qs, 16);
            dst[qs_len + (size_t)(row * n_blocks + blk)] = b->e;
        }
    }
}

// Host repack of one IQ4_NL matrix: canonical block_iq4_nl (18 B: f16 d, 16 nibble bytes) to de-aliased nibble rows + an f16 plane.
// Padding stays zero, and a zero f16 is a zero scale, so padding can never leak into a sum.
void repack_iq4nl_host(const block_iq4_nl * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    GGML_ASSERT(ne0 % 32 == 0);
    const int64_t n_blocks = ne0 / 32;
    const int64_t qs_str   = repack_qs_row_stride(GGML_TYPE_IQ4_NL, ne0);
    const size_t  qs_len   = (size_t) ne1 * qs_str;

    memset(dst, 0, qs_len + (size_t) ne1 * n_blocks * 2);

    for (int64_t row = 0; row < ne1; row++) {
        for (int64_t blk = 0; blk < n_blocks; blk++) {
            const block_iq4_nl * b = &blocks[row * n_blocks + blk];
            memcpy(dst + (size_t) row * qs_str + (size_t) blk * 16, b->qs, 16);
            memcpy(dst + qs_len + (size_t)(row * n_blocks + blk) * 2, &b->d, 2);
        }
    }
}

// Host repack of one Q6_K matrix.
// Canonical block_q6_K packs 256 values as ql[128] low nibbles + qh[64] 2-bit highs in an interleaved order, 16 int8 scales (one per 16 values) and one f16 d.
// Planes per 32-value sub-block: 16 B packed lows on the de-aliased row stride (byte j = lows of values j and j+16), 8 B packed highs (byte m = 2-bit fields of values 4m..4m+3, low half first), a 2-byte int8 scale pair, and the f16 d per super-block in its own plane.
void repack_q6k_host(const block_q6_K * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    GGML_ASSERT(ne0 % 256 == 0);
    const int64_t n_sub  = ne0 / 32;
    const int64_t nds    = (n_sub + 7) >> 3;
    const int64_t qs_str = repack_qs_row_stride(GGML_TYPE_Q6_K, ne0);
    const size_t  hoff   = (size_t) ne1 * qs_str;
    const size_t  soff   = hoff + (size_t) ne1 * n_sub * 8;
    const size_t  doff   = soff + (size_t) ne1 * n_sub * 2;

    memset(dst, 0, doff + (size_t) ne1 * nds * 2);

    for (int64_t row = 0; row < ne1; row++) {
        const block_q6_K * brow = blocks + row * (ne0 / 256);
        for (int64_t sb = 0; sb < n_sub; sb++) {
            const block_q6_K * b = &brow[sb >> 3];
            const int k = (int)(sb & 7);
            uint8_t raw[32];
            for (int i = 0; i < 32; i++) {
                const int v = k * 32 + i;
                const int h = v >> 7;
                const int l = v & 127;
                const uint8_t * ql = b->ql + 64 * h;
                const uint8_t * qh = b->qh + 32 * h;
                uint8_t r6;
                if      (l <  32) { r6 = (ql[l]      & 0xF) | (((qh[l]      >> 0) & 3) << 4); }
                else if (l <  64) { r6 = (ql[l]      & 0xF) | (((qh[l - 32] >> 2) & 3) << 4); }
                else if (l <  96) { r6 = (ql[l - 64] >>  4) | (((qh[l - 64] >> 4) & 3) << 4); }
                else              { r6 = (ql[l - 64] >>  4) | (((qh[l - 96] >> 6) & 3) << 4); }
                raw[i] = r6;
            }
            const size_t wi = (size_t) row * n_sub + sb;
            uint8_t * lows  = dst + (size_t) row * qs_str + (size_t) sb * 16;
            uint8_t * highs = dst + hoff + wi * 8;
            for (int j = 0; j < 16; j++) {
                lows[j] = (uint8_t)((raw[j] & 0xF) | ((raw[j + 16] & 0xF) << 4));
            }
            for (int m = 0; m < 8; m++) {
                const int base = (m & 3) * 4 + (m >> 2) * 16;
                highs[m] = (uint8_t)( ((raw[base + 0] >> 4) << 0)
                                    | ((raw[base + 1] >> 4) << 2)
                                    | ((raw[base + 2] >> 4) << 4)
                                    | ((raw[base + 3] >> 4) << 6));
            }
            dst[soff + wi * 2 + 0] = (uint8_t) b->scales[2 * k + 0];
            dst[soff + wi * 2 + 1] = (uint8_t) b->scales[2 * k + 1];
            if (k == 0) {
                memcpy(dst + doff + ((size_t) row * nds + (sb >> 3)) * 2, &b->d, 2);
            }
        }
    }
}

// Host repack of one Q4_K matrix.
// Canonical block_q4_K: half d, half dmin, the 12-byte packed scale/min record, then qs[128] where sub-block 2j lives in the low nibbles of bytes 32j..32j+31 and sub-block 2j+1 in their high nibbles.
// Planes: 16 B packed nibbles per sub-block on the de-aliased row stride (byte i = value i | value (16+i) << 4), the scale record verbatim per super-block, half2 {d, dmin}.
void repack_q4k_host(const block_q4_K * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    GGML_ASSERT(ne0 % 256 == 0);
    const int64_t n_sub  = ne0 / 32;
    const int64_t nsb    = (n_sub + 7) >> 3;
    const int64_t qs_str = repack_qs_row_stride(GGML_TYPE_Q4_K, ne0);
    const size_t  soff   = (size_t) ne1 * qs_str;
    const size_t  doff   = soff + (size_t) ne1 * nsb * 12;

    memset(dst, 0, doff + (size_t) ne1 * nsb * 4);

    for (int64_t row = 0; row < ne1; row++) {
        const block_q4_K * brow = blocks + row * (ne0 / 256);
        for (int64_t sb = 0; sb < n_sub; sb++) {
            const block_q4_K * b = &brow[sb >> 3];
            const int k = (int)(sb & 7);
            const uint8_t * q = b->qs + 32 * (k >> 1);
            const int shift = (k & 1) * 4;
            uint8_t * lows = dst + (size_t) row * qs_str + (size_t) sb * 16;
            for (int j = 0; j < 16; j++) {
                const uint8_t v0 = (q[j]      >> shift) & 0xF;
                const uint8_t v1 = (q[j + 16] >> shift) & 0xF;
                lows[j] = (uint8_t)(v0 | (v1 << 4));
            }
            if (k == 0) {
                const size_t sbk = (size_t) row * nsb + (sb >> 3);
                memcpy(dst + soff + sbk * 12, b->scales, 12);
                memcpy(dst + doff + sbk * 4, b, 4);
            }
        }
    }
}

// Host repack of one Q5_K matrix: Q4_K's planes plus the fifth bit.
// Canonical block_q5_K keeps it in qh[32], bit (2j + (k & 1)) of qh[i] for value i of sub-block k = 2j + (k & 1).
// The plane stores one 32-bit word per sub-block, bit i = value i.
void repack_q5k_host(const block_q5_K * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    GGML_ASSERT(ne0 % 256 == 0);
    const int64_t n_sub  = ne0 / 32;
    const int64_t nsb    = (n_sub + 7) >> 3;
    const int64_t qs_str = repack_qs_row_stride(GGML_TYPE_Q5_K, ne0);
    const size_t  hoff   = (size_t) ne1 * qs_str;
    const size_t  soff   = hoff + (size_t) ne1 * n_sub * 4;
    const size_t  doff   = soff + (size_t) ne1 * nsb * 12;

    memset(dst, 0, doff + (size_t) ne1 * nsb * 4);

    for (int64_t row = 0; row < ne1; row++) {
        const block_q5_K * brow = blocks + row * (ne0 / 256);
        for (int64_t sb = 0; sb < n_sub; sb++) {
            const block_q5_K * b = &brow[sb >> 3];
            const int k = (int)(sb & 7);
            const uint8_t * q = b->qs + 32 * (k >> 1);
            const int shift = (k & 1) * 4;
            const int hbit  = 2 * (k >> 1) + (k & 1);
            const size_t wi = (size_t) row * n_sub + sb;
            uint8_t * lows = dst + (size_t) row * qs_str + (size_t) sb * 16;
            uint32_t h = 0;
            for (int j = 0; j < 16; j++) {
                const uint8_t v0 = (q[j]      >> shift) & 0xF;
                const uint8_t v1 = (q[j + 16] >> shift) & 0xF;
                lows[j] = (uint8_t)(v0 | (v1 << 4));
            }
            for (int i = 0; i < 32; i++) {
                h |= (uint32_t)((b->qh[i] >> hbit) & 1) << i;
            }
            memcpy(dst + hoff + wi * 4, &h, 4);
            if (k == 0) {
                const size_t sbk = (size_t) row * nsb + (sb >> 3);
                memcpy(dst + soff + sbk * 12, b->scales, 12);
                memcpy(dst + doff + sbk * 4, b, 4);
            }
        }
    }
}

// Host repack of one Q5_1 matrix: the canonical nibble bytes and fifth-bit word per block are already in plane order, so the three planes are copies.
void repack_q51_host(const block_q5_1 * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    GGML_ASSERT(ne0 % 32 == 0);
    const int64_t n_sub  = ne0 / 32;
    const int64_t qs_str = repack_qs_row_stride(GGML_TYPE_Q5_1, ne0);
    const size_t  hoff   = (size_t) ne1 * qs_str;
    const size_t  doff   = hoff + (size_t) ne1 * n_sub * 4;

    memset(dst, 0, doff + (size_t) ne1 * n_sub * 4);

    for (int64_t row = 0; row < ne1; row++) {
        const block_q5_1 * brow = blocks + row * n_sub;
        for (int64_t sb = 0; sb < n_sub; sb++) {
            const block_q5_1 * b = &brow[sb];
            const size_t wi = (size_t) row * n_sub + sb;
            memcpy(dst + (size_t) row * qs_str + (size_t) sb * 16, b->qs, 16);
            memcpy(dst + hoff + wi * 4, b->qh, 4);
            memcpy(dst + doff + wi * 4, &b->dm, 4);
        }
    }
}

void repack_host(ggml_type type, const void * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1) {
    switch (type) {
        case GGML_TYPE_Q8_0:
            repack_q8_0_host((const block_q8_0 *) blocks, dst, ne0, ne1);
            break;
        case GGML_TYPE_CT_INT4:
            repack_ct_int4_host((const block_ct_int4 *) blocks, dst, ne0, ne1);
            break;
        case GGML_TYPE_MXFP4:
            repack_mxfp4_host((const block_mxfp4 *) blocks, dst, ne0, ne1);
            break;
        case GGML_TYPE_IQ4_NL:
            repack_iq4nl_host((const block_iq4_nl *) blocks, dst, ne0, ne1);
            break;
        case GGML_TYPE_Q6_K:
            repack_q6k_host((const block_q6_K *) blocks, dst, ne0, ne1);
            break;
        case GGML_TYPE_Q4_K:
            repack_q4k_host((const block_q4_K *) blocks, dst, ne0, ne1);
            break;
        case GGML_TYPE_Q5_K:
            repack_q5k_host((const block_q5_K *) blocks, dst, ne0, ne1);
            break;
        case GGML_TYPE_Q5_1:
            repack_q51_host((const block_q5_1 *) blocks, dst, ne0, ne1);
            break;
        default:
            GGML_ABORT("unsupported repack type");
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
const uint8_t * repack_view_get_cached(
        const ggml_tensor * view, const ggml_tensor * base,
        cudaStream_t stream) {
    // the K-quant multi-plane layouts have no view splice: supports() refuses their views
    GGML_ASSERT(base->type != GGML_TYPE_Q6_K && base->type != GGML_TYPE_Q4_K && base->type != GGML_TYPE_Q5_K && base->type != GGML_TYPE_Q5_1 && "repack: multi-plane views are not spliced");
    const int64_t ne0_v = view->ne[0];
    const int64_t ne1_v = view->ne[1];
    const int64_t ne2_v = view->ne[2];
    const int64_t ne0_b = base->ne[0];
    const int64_t ne1_b = base->ne[1];
    const int64_t ne2_b = base->ne[2];
    GGML_ASSERT(ne0_v == ne0_b);
    GGML_ASSERT(base->view_src == nullptr);

    // Per-type strides: MXFP4 rows carry packed nibbles (ne0/2 B) with 1 B
    // e8m0 scales, Q8_0 rows carry ne0 B with 2 B f16 scales.
    const int64_t qs_str  = repack_qs_row_stride(base->type, ne0_v);
    const int64_t sc_row  = repack_scale_row_bytes(base->type, ne0_v);
    const uint8_t * base_ptr = (const uint8_t *) base->data;
    const uint8_t * view_ptr = (const uint8_t *) view->data;

    const int64_t row_start = (int64_t)(view_ptr - base_ptr) / qs_str;
    GGML_ASSERT((view_ptr - base_ptr) % qs_str == 0);

    // Copy every expert in the view (ne1_v x ne2_v), not just ne1_v rows.
    const size_t qs_size = (size_t) ne1_v * ne2_v * qs_str;
    const size_t sc_size = (size_t) ne1_v * ne2_v * sc_row;
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
                                       + (size_t) row_start * sc_row;
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
