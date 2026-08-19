// Shared layout helpers and device-side data types for the Q8_0 repacked-weight path.
#pragma once

#include "../common.cuh"
#include "../mmq.cuh"

#include <cstddef>

#define MMQ_RP_Q8_BK 4
#define MMQ_RP_Q8_TN 2
#define MMQ_RP_Q8_BM 64
#define MMQ_RP_Q8_NROW_LANES 4
#define MMQ_RP_Q8_MOE_W32_MAX_TOKENS 1024

template <typename T>
static __host__ __device__ inline T repack_nsp(const T ne0) {
    const T n_sub = ne0 / 32;
    return (n_sub & (n_sub - 1)) == 0 ? n_sub + 1 : n_sub;
}

static inline size_t repack_gcn_nbytes(const ggml_type type, const int64_t ne0, const int64_t ne1) {
    GGML_ASSERT(ne0 % 32 == 0);
    const int64_t nsp      = repack_nsp(ne0);
    switch (type) {
        case GGML_TYPE_Q8_0: return (size_t) ne1 * nsp * 34;
        default:             GGML_ABORT("unsupported repack type");
    }
}

template <int CW>
static __device__ __forceinline__ int sX_swizzle(int lr) {
    if constexpr (CW == 64) {
        const int n  = lr >> 6;
        int       tx = lr & 63;
        tx ^= (tx >> 5) << 4;
        return (n << 6) | tx;
    } else {
        const int n  = lr >> 5;
        int       tx = lr & 31;
        tx ^= (tx >> 4) << 3;
        return (n << 5) | tx;
    }
}

struct rp_x_sub {
    uint4 q0, q1;
    float d;
};

struct block_q8_1_mmq_h {
    float  d4[4];
    int8_t qs[QK8_1_MMQ];
};

static_assert(sizeof(block_q8_1_mmq_h) == sizeof(block_q8_1_mmq),
              "Unexpected block_q8_1_mmq_h size");
static_assert(offsetof(block_q8_1_mmq_h, d4) == offsetof(block_q8_1_mmq, d4),
              "block_q8_1_mmq_h d4 offset mismatch");
static_assert(offsetof(block_q8_1_mmq_h, qs) == offsetof(block_q8_1_mmq, qs),
              "block_q8_1_mmq_h qs offset mismatch");

struct sXq_row_q8 {
    uint4 q[MMQ_RP_Q8_BK][2];
    uint4 pad;
};

static_assert(sizeof(sXq_row_q8) == (2 * MMQ_RP_Q8_BK + 1) * 16,
              "unexpected sXq row size");

static __device__ __forceinline__ uint4 rp_ldcs_u4(const uint4 * __restrict__ p) {
    return *p;
}

#if defined(GGML_USE_HIP) && defined(__gfx906__)

#define RP_DPP_ADD(name, nop, dpp_ctrl)                                      \
    static __device__ __forceinline__ float name(const float x) {            \
        float r;                                                             \
        asm volatile(                                                        \
            nop                                                              \
            "v_add_f32_dpp %0, %1, %1 " dpp_ctrl " row_mask:0xf bank_mask:0xf" \
            : "=v"(r) : "v"(x) : "memory");                                  \
        return r;                                                            \
    }

RP_DPP_ADD(rp_dpp_add_xor1, "s_nop 4\n", "quad_perm:[1,0,3,2]")
RP_DPP_ADD(rp_dpp_add_xor2, "s_nop 1\n", "quad_perm:[2,3,0,1]")
RP_DPP_ADD(rp_dpp_add_xor8, "s_nop 1\n", "row_ror:8")

#undef RP_DPP_ADD

static __device__ __forceinline__ float rp_dpp_xfer_xor4(const float x) {
    int d;
    asm volatile("v_mov_b32 %0, %1\n"
                 "s_nop 1\n"
                 "v_mov_b32_dpp %0, %1 row_shl:4 row_mask:0xf bank_mask:0x5\n"
                 "v_mov_b32_dpp %0, %1 row_shr:4 row_mask:0xf bank_mask:0xa\n"
                 : "=v"(d) : "v"(__float_as_int(x)) : "memory");
    return __int_as_float(d);
}

static __device__ __forceinline__ float rp_dpp_xfer_xor16(const float x) {
    int d;
    asm volatile("ds_swizzle_b32 %0, %1 offset:swizzle(SWAP,16)\n"
                 "s_waitcnt lgkmcnt(0)\n"
                 : "=v"(d) : "v"(__float_as_int(x)) : "memory");
    return __int_as_float(d);
}

template <int width>
static __device__ __forceinline__ float rp_warp_reduce_sum(const float x) {
    static_assert(width >= 1 && (width & (width - 1)) == 0);
    float r = x;
    if constexpr (width >=  2) { r = rp_dpp_add_xor1(r); }
    if constexpr (width >=  4) { r = rp_dpp_add_xor2(r); }
    if constexpr (width >=  8) { r += rp_dpp_xfer_xor4(r); }
    if constexpr (width >= 16) { r = rp_dpp_add_xor8(r); }
    if constexpr (width >= 32) { r += rp_dpp_xfer_xor16(r); }
    if constexpr (width >= 64) { r += __shfl_xor_sync(0xffffffff, r, 32, 64); }
    return r;
}

#else

template <int width>
static __device__ __forceinline__ float rp_warp_reduce_sum(const float x) {
    return warp_reduce_sum<width>(x);
}

#endif

__device__ __forceinline__ rp_x_sub rp_x_sub_from_mmq_group(
        const block_q8_1_mmq_h * __restrict__ group, const uint32_t col, const uint32_t lk) {
    const block_q8_1_mmq_h & m = group[col];
    const uint4           * mq = reinterpret_cast<const uint4 *>(m.qs + lk * QK8_1);
    rp_x_sub out;
    out.q0 = mq[0];
    out.q1 = mq[1];
    out.d  = m.d4[lk];
    return out;
}

void repack_q8_0_host(const block_q8_0 * blocks, uint8_t * dst, const int64_t ne0, const int64_t ne1);

const uint8_t * repack_q8_0_view_get_cached(
        const ggml_tensor * view, const ggml_tensor * base, cudaStream_t stream);
