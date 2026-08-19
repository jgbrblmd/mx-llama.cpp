// Device kernels and helpers for the Q8_0 repacked-weight path (AMD GCN only). MMV
// and GEMM are templated on HAS_IDS so dense and MoE host paths share identical device
// code. All __global__ kernels and __device__ functions use
// #if defined(GGML_USE_HIP) && defined(__gfx906__) bodies; elsewhere NO_DEVICE_CODE stubs.
#pragma once

#include "../common.cuh"
#include "../mmq.cuh"
#include "../unary.cuh"
#include "repack-common.cuh"

struct repack_tile_meta {
    uint32_t expert;
    uint32_t a_base;
};

template <int BN>
static __global__ void repack_tile_off(
        const int32_t * __restrict__ expert_bounds, int32_t * __restrict__ tile_off,
        repack_tile_meta * __restrict__ tile_meta, const int n_expert) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    int acc = 0;
    tile_off[0] = 0;
    for (int e = 0; e < n_expert; e++) {
        const int cnt = expert_bounds[e + 1] - expert_bounds[e];
        const int nt  = (cnt + BN - 1) / BN;
        for (int t = 0; t < nt; t++) {
            tile_meta[acc + t].expert = (uint32_t) e;
            tile_meta[acc + t].a_base = (uint32_t) expert_bounds[e] + (uint32_t) t * BN;
        }
        acc += nt;
        tile_off[e + 1] = acc;
    }
#else
    GGML_UNUSED_VARS(expert_bounds, tile_off, tile_meta, n_expert);
    NO_DEVICE_CODE;
#endif
}

static __device__ __forceinline__ int mmvq_dp4a_u4(
        const uint4 w, const int x0, const int x1, const int x2, const int x3) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    int a = ggml_cuda_dp4a((int) w.x, x0, 0);
    int b = ggml_cuda_dp4a((int) w.z, x2, 0);
    a = ggml_cuda_dp4a((int) w.y, x1, a);
    b = ggml_cuda_dp4a((int) w.w, x3, b);
    return a + b;
#else
    GGML_UNUSED_VARS(w, x0, x1, x2, x3);
    return 0;
#endif
}

// Fused-FFN epilogue shared by the dense and MoE MMV paths: up result + x_bias,
// gate result + gate_bias, then the GLU op.
static __device__ __forceinline__ float rp_mmv_fusion_epilogue(
        const float up, const float gate, const float * x_bias, const float * gate_bias,
        const uint32_t bias_idx, const ggml_glu_op glu_op, const bool use_gate) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    float result = up;
    if (x_bias != nullptr) {
        result += x_bias[bias_idx];
    }
    if (use_gate) {
        float gate_value = gate;
        if (gate_bias != nullptr) {
            gate_value += gate_bias[bias_idx];
        }
        switch (glu_op) {
            case GGML_GLU_OP_SWIGLU:     result *= ggml_cuda_op_silu_single(gate_value); break;
            case GGML_GLU_OP_GEGLU:      result *= ggml_cuda_op_gelu_single(gate_value); break;
            case GGML_GLU_OP_SWIGLU_OAI: result = ggml_cuda_op_swiglu_oai_single(gate_value, result); break;
            default:                     result *= gate_value; break;
        }
    }
    return result;
#else
    GGML_UNUSED_VARS(gate, x_bias, gate_bias, bias_idx, glu_op, use_gate);
    return up;
#endif
}

template <int ROWS, int NWAVES, bool HAS_IDS, int LANES = 64, bool HAS_FUSION = false>
static __global__ void mul_mat_vec_q8_0_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const uint32_t n_expert,
        const uint32_t nchannels_y, const size_t expert_stride,
        const uint32_t xs_id, const uint32_t dst_s1,
        const uint8_t * __restrict__ wbase_gate, const float * __restrict__ x_bias,
        const float * __restrict__ gate_bias, const ggml_glu_op glu_op) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    constexpr int NWARPS = (NWAVES * 64) / LANES;
    constexpr int WPR = NWARPS / ROWS;
    static_assert((NWAVES * 64) % (ROWS * LANES) == 0, "NWAVES*64 must be a multiple of ROWS*LANES");
    static_assert(WPR >= 1 && (WPR & (WPR - 1)) == 0, "WPR must be a power of two");

    if constexpr (!HAS_FUSION) {
        GGML_UNUSED_VARS(wbase_gate, x_bias, gate_bias, glu_op);
    }

    [[maybe_unused]] const bool use_gate = wbase_gate != nullptr;

    [[maybe_unused]] uint32_t expert = 0;
    if constexpr (HAS_IDS) {
        const uint32_t a = blockIdx.y;
        const uint32_t e = (uint32_t) ids_src1[a];
        expert = e;
        wbase += e * expert_stride;
        xq    += (size_t)(a % nchannels_y) * xs_id;
        y     += (size_t) a * dst_s1;
        GGML_UNUSED_VARS(ids_dst, expert_bounds, n_expert);
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, n_expert, nchannels_y, expert_stride, xs_id, dst_s1);
    }
    const uint32_t n_blocks = ne0 >> 5;
    const uint32_t nsp      = repack_nsp(ne0);

    const int warp_id   = threadIdx.x / LANES;
    const int lane      = threadIdx.x % LANES;
    const int row_local = warp_id / WPR;
    const int k_slice   = warp_id % WPR;
    const int row       = blockIdx.x * ROWS + row_local;

    const uint32_t wrow     = min((uint32_t) row, ne1 - 1);
    const uint4    * w_row  = reinterpret_cast<const uint4 *>(wbase) + (size_t) wrow * nsp * 2;
    const uint16_t * d_row  = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 32) + (size_t) wrow * nsp;
    const uint16_t rmask  = (row < (int) ne1) ? (uint16_t) 0xffffu : (uint16_t) 0x0000u;

    [[maybe_unused]] const uint4 * w_row_gate = nullptr;
    [[maybe_unused]] const uint16_t * d_row_gate = nullptr;
    if constexpr (HAS_FUSION) {
        if (use_gate) {
            w_row_gate = reinterpret_cast<const uint4 *>(wbase_gate) + (size_t) wrow * nsp * 2;
            d_row_gate = reinterpret_cast<const uint16_t *>(
                wbase_gate + (size_t) ne1 * nsp * 32) + (size_t) wrow * nsp;
        }
    }

    [[maybe_unused]] uint32_t bias_row = 0;
    if constexpr (HAS_FUSION) {
        if constexpr (HAS_IDS) {
            bias_row = expert * ne1 + (uint32_t) row;
        } else {
            bias_row = (uint32_t) row;
        }
    }

    float acc = 0.0f;
    [[maybe_unused]] float acc_gate = 0.0f;

    if constexpr (LANES < 64) {
        const uint32_t n_half = n_blocks * 2;
        const uint32_t hb_lo  = (n_half * k_slice) / WPR;
        const uint32_t hb_hi  = (n_half * (k_slice + 1)) / WPR;

        const uint32_t hb0 = hb_lo + lane;
        const uint32_t half = hb0 & 1;
        const uint4 * wp = w_row + hb0;
        const block_q8_1 * xp = xq + (hb0 >> 1);
        const uint16_t * dp = d_row + (hb0 >> 1);

        [[maybe_unused]] const uint4 * wpg = nullptr;
        [[maybe_unused]] const uint16_t * dpg = nullptr;
        if constexpr (HAS_FUSION) {
            if (use_gate) {
                wpg = w_row_gate + hb0;
                dpg = d_row_gate + (hb0 >> 1);
            }
        }

        uint32_t iters = 0;
        for (uint32_t hb = hb0; hb + LANES < hb_hi; hb += 2 * LANES) {
            const float dx0 = __low2float(xp->ds);
            const int * xq32_0 = reinterpret_cast<const int *>(xp->qs) + half * 4;
            const int x00 = xq32_0[0], x01 = xq32_0[1], x02 = xq32_0[2], x03 = xq32_0[3];
            const uint4 wv0 = *wp;
            const uint16_t db0 = *dp & rmask;

            const float dx1 = __low2float((xp + LANES / 2)->ds);
            const int * xq32_1 = reinterpret_cast<const int *>((xp + LANES / 2)->qs) + half * 4;
            const int x10 = xq32_1[0], x11 = xq32_1[1], x12 = xq32_1[2], x13 = xq32_1[3];
            const uint4 wv1 = *(wp + LANES);
            const uint16_t db1 = *(dp + LANES / 2) & rmask;

            const float dw0 = __half2float(*reinterpret_cast<const __half *>(&db0));
            int idot0 = 0;
            idot0 = ggml_cuda_dp4a((int) wv0.x, x00, idot0);
            idot0 = ggml_cuda_dp4a((int) wv0.y, x01, idot0);
            idot0 = ggml_cuda_dp4a((int) wv0.z, x02, idot0);
            idot0 = ggml_cuda_dp4a((int) wv0.w, x03, idot0);
            acc += dw0 * dx0 * (float) idot0;

            const float dw1 = __half2float(*reinterpret_cast<const __half *>(&db1));
            int idot1 = 0;
            idot1 = ggml_cuda_dp4a((int) wv1.x, x10, idot1);
            idot1 = ggml_cuda_dp4a((int) wv1.y, x11, idot1);
            idot1 = ggml_cuda_dp4a((int) wv1.z, x12, idot1);
            idot1 = ggml_cuda_dp4a((int) wv1.w, x13, idot1);
            acc += dw1 * dx1 * (float) idot1;

            if constexpr (HAS_FUSION) {
                if (use_gate) {
                    const uint4 wgv0 = *wpg;
                    const uint16_t dgb0 = *dpg & rmask;
                    const float dwg0 = __half2float(*reinterpret_cast<const __half *>(&dgb0));
                    acc_gate += dwg0 * dx0 * (float) mmvq_dp4a_u4(wgv0, x00, x01, x02, x03);

                    const uint4 wgv1 = *(wpg + LANES);
                    const uint16_t dgb1 = *(dpg + LANES / 2) & rmask;
                    const float dwg1 = __half2float(*reinterpret_cast<const __half *>(&dgb1));
                    acc_gate += dwg1 * dx1 * (float) mmvq_dp4a_u4(wgv1, x10, x11, x12, x13);
                }
            }

            wp += 2 * LANES;
            xp += LANES;
            dp += LANES;
            if constexpr (HAS_FUSION) {
                if (use_gate) {
                    wpg += 2 * LANES;
                    dpg += LANES;
                }
            }
            iters += 2;
        }
        if (hb0 + iters * LANES < hb_hi) {
            const float dx = __low2float(xp->ds);
            const int * xq32 = reinterpret_cast<const int *>(xp->qs) + half * 4;
            const int x0 = xq32[0], x1 = xq32[1], x2 = xq32[2], x3 = xq32[3];
            const uint4 wv = *wp;
            const uint16_t db = *dp & rmask;
            const float dw = __half2float(*reinterpret_cast<const __half *>(&db));
            int idot = 0;
            idot = ggml_cuda_dp4a((int) wv.x, x0, idot);
            idot = ggml_cuda_dp4a((int) wv.y, x1, idot);
            idot = ggml_cuda_dp4a((int) wv.z, x2, idot);
            idot = ggml_cuda_dp4a((int) wv.w, x3, idot);
            acc += dw * dx * (float) idot;

            if constexpr (HAS_FUSION) {
                if (use_gate) {
                    const uint4 wgv = *wpg;
                    const uint16_t dgb = *dpg & rmask;
                    const float dwg = __half2float(*reinterpret_cast<const __half *>(&dgb));
                    acc_gate += dwg * dx * (float) mmvq_dp4a_u4(wgv, x0, x1, x2, x3);
                }
            }
        }
    } else {
        const uint32_t n_half = n_blocks * 2;
        const uint32_t hb_lo  = (n_half * k_slice) / WPR;
        const uint32_t hb_hi  = (n_half * (k_slice + 1)) / WPR;

        uint32_t base = hb_lo;
        for (; base + 2u * (uint32_t) LANES <= hb_hi; base += 2u * (uint32_t) LANES) {
            const uint32_t hb    = base + (uint32_t) lane;
            const uint32_t sb0   = hb >> 1;
            const uint32_t half0 = hb & 1;
            const uint32_t hb1   = hb + LANES;
            const uint32_t sb1   = hb1 >> 1;
            const uint32_t half1 = hb1 & 1;

            const block_q8_1 * xb0 = xq + sb0;
            const float dx0 = __low2float(xb0->ds);
            const int * xq32_0 = reinterpret_cast<const int *>(xb0->qs) + half0 * 4;
            const int x00 = xq32_0[0], x01 = xq32_0[1], x02 = xq32_0[2], x03 = xq32_0[3];
            const uint4 wv0 = w_row[hb];
            const uint16_t db0 = d_row[sb0] & rmask;

            const block_q8_1 * xb1 = xq + sb1;
            const float dx1 = __low2float(xb1->ds);
            const int * xq32_1 = reinterpret_cast<const int *>(xb1->qs) + half1 * 4;
            const int x10 = xq32_1[0], x11 = xq32_1[1], x12 = xq32_1[2], x13 = xq32_1[3];
            const uint4 wv1 = w_row[hb1];
            const uint16_t db1 = d_row[sb1] & rmask;

            const float dw0 = __half2float(*reinterpret_cast<const __half *>(&db0));
            const int idot0 = mmvq_dp4a_u4(wv0, x00, x01, x02, x03);
            acc += dw0 * dx0 * (float) idot0;

            const float dw1 = __half2float(*reinterpret_cast<const __half *>(&db1));
            const int idot1 = mmvq_dp4a_u4(wv1, x10, x11, x12, x13);
            acc += dw1 * dx1 * (float) idot1;

            if constexpr (HAS_FUSION) {
                if (use_gate) {
                    const uint4 wgv0 = w_row_gate[hb];
                    const uint16_t dgb0 = d_row_gate[sb0] & rmask;
                    const float dwg0 = __half2float(*reinterpret_cast<const __half *>(&dgb0));
                    acc_gate += dwg0 * dx0 * (float) mmvq_dp4a_u4(wgv0, x00, x01, x02, x03);

                    const uint4 wgv1 = w_row_gate[hb1];
                    const uint16_t dgb1 = d_row_gate[sb1] & rmask;
                    const float dwg1 = __half2float(*reinterpret_cast<const __half *>(&dgb1));
                    acc_gate += dwg1 * dx1 * (float) mmvq_dp4a_u4(wgv1, x10, x11, x12, x13);
                }
            }
        }

#pragma unroll
        for (int part = 0; part < 2; ++part) {
            const uint32_t hb = base + (uint32_t) (part * LANES + lane);
            if (hb < hb_hi) {
                const uint32_t sb   = hb >> 1;
                const uint32_t half = hb & 1;
                const block_q8_1 * xb = xq + sb;
                const float dx = __low2float(xb->ds);
                const int * xq32 = reinterpret_cast<const int *>(xb->qs) + half * 4;
                const int x0 = xq32[0], x1 = xq32[1], x2 = xq32[2], x3 = xq32[3];
                const uint4 wv = w_row[hb];
                const uint16_t db = d_row[sb] & rmask;
                const float dw = __half2float(*reinterpret_cast<const __half *>(&db));
                const int idot = mmvq_dp4a_u4(wv, x0, x1, x2, x3);
                acc += dw * dx * (float) idot;

                if constexpr (HAS_FUSION) {
                    if (use_gate) {
                        const uint4 wgv = w_row_gate[hb];
                        const uint16_t dgb = d_row_gate[sb] & rmask;
                        const float dwg = __half2float(*reinterpret_cast<const __half *>(&dgb));
                        acc_gate += dwg * dx * (float) mmvq_dp4a_u4(wgv, x0, x1, x2, x3);
                    }
                }
            }
        }
    }

    if constexpr (WPR > 1) {
        __shared__ float s_part[ROWS][WPR];
        [[maybe_unused]] __shared__ float s_part_gate[ROWS][WPR];
        const float wsum = rp_warp_reduce_sum<LANES>(acc);
        if (lane == 0) {
            s_part[row_local][k_slice] = wsum;
        }
        if constexpr (HAS_FUSION) {
            const float wsum_gate = rp_warp_reduce_sum<LANES>(acc_gate);
            if (lane == 0) {
                s_part_gate[row_local][k_slice] = wsum_gate;
            }
        }
        __syncthreads();
        if (k_slice == 0 && lane == 0) {
            float sum = 0.0f;
#pragma unroll
            for (int w = 0; w < WPR; w++) {
                sum += s_part[row_local][w];
            }
            if (row < (int) ne1) {
                if constexpr (HAS_FUSION) {
                    float gate_sum = 0.0f;
#pragma unroll
                    for (int w = 0; w < WPR; w++) {
                        gate_sum += s_part_gate[row_local][w];
                    }
                    y[row] = rp_mmv_fusion_epilogue(sum, gate_sum, x_bias, gate_bias, bias_row, glu_op, use_gate);
                } else {
                    y[row] = sum;
                }
            }
        }
    } else {
        const float a = rp_warp_reduce_sum<LANES>(acc);
        if constexpr (HAS_FUSION) {
            const float g = rp_warp_reduce_sum<LANES>(acc_gate);
            if (lane == 0 && row < (int) ne1) {
                y[row] = rp_mmv_fusion_epilogue(a, g, x_bias, gate_bias, bias_row, glu_op, use_gate);
            }
        } else {
            if (lane == 0 && row < (int) ne1) {
                y[row] = a;
            }
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, ids_src1, ids_dst, expert_bounds, n_expert, nchannels_y, expert_stride, xs_id, dst_s1, wbase_gate, x_bias, gate_bias, glu_op);
    NO_DEVICE_CODE;
#endif
}

template <bool HAS_IDS, int CW, int TN_, int NRL>
static __device__ void mmq_gemm_q8_0_repacked_impl(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t n_tok,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ tile_off,
        const repack_tile_meta * __restrict__ tile_meta,
        const uint32_t n_expert, const size_t expert_stride, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    const int t  = threadIdx.x + threadIdx.y * blockDim.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const uint32_t row0 = blockIdx.x * MMQ_RP_Q8_BM;
    constexpr uint32_t BN = CW * TN_;
    uint32_t tok0 = blockIdx.y * BN;
    uint32_t a_base = 0, a_end = 0;
    if constexpr (HAS_IDS) {
        if (blockIdx.y >= (uint32_t) tile_off[n_expert]) {
            return;
        }
        const repack_tile_meta meta = tile_meta[blockIdx.y];
        const uint32_t e = meta.expert;
        a_base = meta.a_base;
        a_end  = (uint32_t) expert_bounds[e + 1];
        wbase += e * expert_stride;
        tok0   = 0;
    } else {
        GGML_UNUSED_VARS(ids_src1, ids_dst, expert_bounds, tile_off, tile_meta, n_expert, expert_stride);
    }

    const uint32_t n_sub = ne0 >> 5;
    const uint32_t nsp   = repack_nsp(ne0);
    const uint4    * __restrict__ qsp = reinterpret_cast<const uint4 *>(wbase);
    const uint16_t * __restrict__ dp  = reinterpret_cast<const uint16_t *>(
        wbase + (size_t) ne1 * nsp * 32);
    const block_q8_1_mmq_h * __restrict__ xmmq = reinterpret_cast<const block_q8_1_mmq_h *>(xq);

    const block_q8_1_mmq_h * x_group = xmmq;

    __shared__ uint4    sW_lo[MMQ_RP_Q8_BM][MMQ_RP_Q8_BK];
    __shared__ uint4    sW_hi[MMQ_RP_Q8_BM][MMQ_RP_Q8_BK];
    __shared__ uint16_t sWdh[MMQ_RP_Q8_BK][MMQ_RP_Q8_BM];
    __shared__ sXq_row_q8 sXq[BN];
    __shared__ float    sXd[BN][MMQ_RP_Q8_BK + 1];
    __shared__ uint32_t sCol[BN];

    constexpr int NROW = MMQ_RP_Q8_BM / NRL;
    float acc[NROW][TN_] = {};

    constexpr int NTHREADS = CW * NRL;
    constexpr int W_ELM = MMQ_RP_Q8_BM * MMQ_RP_Q8_BK;
    constexpr int X_ELM = BN * MMQ_RP_Q8_BK;
    constexpr int W_EPT = (W_ELM + NTHREADS - 1) / NTHREADS;
    constexpr int X_EPT = (X_ELM + NTHREADS - 1) / NTHREADS;

    const int sxc0 = sX_swizzle<CW>(tx);

    uint4    pw_lo[W_EPT];
    uint4    pw_hi[W_EPT];
    uint16_t pw_d [W_EPT];
    rp_x_sub px   [X_EPT];
    if constexpr (HAS_IDS) {
        for (int i = t; i < (int) BN; i += NTHREADS) {
            const uint32_t a = a_base + i;
            sCol[i] = (a < a_end) ? (uint32_t) ids_src1[a] : 0u;
        }
        __syncthreads();
    }
    // Whole BM x BN x BK tile in-bounds -> use the unchecked loads below.
    const bool full_m = row0 + MMQ_RP_Q8_BM <= ne1;
    bool full_n;
    if constexpr (HAS_IDS) {
        full_n = a_base + BN <= a_end;
    } else {
        full_n = tok0 + BN <= n_tok;
    }
    const uint32_t n_main = n_sub & ~(MMQ_RP_Q8_BK - 1u);
    const bool full_k = n_main == n_sub;
    const bool full_tile = full_m && full_n && full_k &&
                           (NTHREADS <= X_ELM) && (NTHREADS <= W_ELM);


    auto prefetch_w_checked = [&](const uint32_t sb0) {
#pragma unroll
        for (int i = 0; i < W_EPT; i++) {
            const int e = t + i * NTHREADS;
            if constexpr (W_EPT * NTHREADS != W_ELM) {
                if (e >= W_ELM) {
                    break;
                }
            }
            const int      lr   = e / MMQ_RP_Q8_BK;
            const int      lk   = e % MMQ_RP_Q8_BK;
            const uint32_t sb   = sb0 + lk;
            const uint32_t wrow = row0 + lr;
            if (wrow < ne1 && sb < n_sub) {
                const size_t wi = (size_t) wrow * nsp + sb;
                pw_lo[i] = rp_ldcs_u4(qsp + 2 * wi);
                pw_hi[i] = rp_ldcs_u4(qsp + 2 * wi + 1);
                pw_d [i] = dp[wi];
            } else {
                pw_d[i] = 0;
            }
        }
    };

    auto prefetch_x_checked = [&](const uint32_t sb0, const block_q8_1_mmq_h * x_grp) {
#pragma unroll
        for (int i = 0; i < X_EPT; i++) {
            const int e = t + i * NTHREADS;
            if constexpr (X_EPT * NTHREADS != X_ELM) {
                if (e >= X_ELM) {
                    break;
                }
            }
            const int      lr = e / MMQ_RP_Q8_BK;
            const int      lk = e % MMQ_RP_Q8_BK;
            const uint32_t sb = sb0 + lk;
            uint32_t xcol = tok0 + lr;
            bool     xval = xcol < n_tok;
            if constexpr (HAS_IDS) {
                const uint32_t a = a_base + lr;
                xval = a < a_end;
                xcol = xval ? sCol[lr] : 0;
            }
            if (xval && sb < n_sub) {
                px[i] = rp_x_sub_from_mmq_group(x_grp, xcol, lk);
            } else {
                px[i].d = 0.0f;
            }
        }
    };

    auto prefetch_w_full = [&](const uint32_t sb0) {
#pragma unroll
        for (int i = 0; i < W_EPT; i++) {
            const int e = t + i * NTHREADS;
            if constexpr (W_EPT * NTHREADS != W_ELM) {
                if (e >= W_ELM) break;
            }
            const int lr = e / MMQ_RP_Q8_BK;
            const int lk = e % MMQ_RP_Q8_BK;

            const uint32_t wrow = row0 + lr;
            const size_t   wi   = (size_t) wrow * nsp + sb0 + lk;

            pw_lo[i] = rp_ldcs_u4(qsp + 2 * wi);
            pw_hi[i] = rp_ldcs_u4(qsp + 2 * wi + 1);
            pw_d [i] = dp[wi];
        }
    };

    auto prefetch_x_full = [&](const uint32_t sb0, const block_q8_1_mmq_h * x_grp) {
#pragma unroll
        for (int i = 0; i < X_EPT; i++) {
            const int e = t + i * NTHREADS;
            if constexpr (X_EPT * NTHREADS != X_ELM) {
                if (e >= X_ELM) break;
            }
            const int lr = e / MMQ_RP_Q8_BK;
            const int lk = e % MMQ_RP_Q8_BK;

            uint32_t xcol = tok0 + lr;
            if constexpr (HAS_IDS) {
                xcol = sCol[lr];
            }

            px[i] = rp_x_sub_from_mmq_group(x_grp, xcol, lk);
        }
    };

    auto store_stage = [&]() {
#pragma unroll
        for (int i = 0; i < W_EPT; i++) {
            const int e = t + i * NTHREADS;
            if constexpr (W_EPT * NTHREADS != W_ELM) {
                if (e >= W_ELM) {
                    break;
                }
            }
            const int lr = e / MMQ_RP_Q8_BK;
            const int lk = e % MMQ_RP_Q8_BK;
            sW_lo[lr][lk] = pw_lo[i];
            sW_hi[lr][lk] = pw_hi[i];
            sWdh [lk][lr] = pw_d[i];
        }
#pragma unroll
        for (int i = 0; i < X_EPT; i++) {
            const int e = t + i * NTHREADS;
            if constexpr (X_EPT * NTHREADS != X_ELM) {
                if (e >= X_ELM) {
                    break;
                }
            }
            const int lr  = e / MMQ_RP_Q8_BK;
            const int lk  = e % MMQ_RP_Q8_BK;
            const int sXr = sX_swizzle<CW>(lr);

            sXq[sXr].q[lk][0] = px[i].q0;
            sXq[sXr].q[lk][1] = px[i].q1;
            sXd[sXr][lk]    = px[i].d;
        }
    };

    auto compute_stage = [&]() {
        for (int kk = 0; kk < MMQ_RP_Q8_BK; kk++) {
            float dx  [TN_];
            int   xq32[TN_][8];
#pragma unroll
            for (int n = 0; n < TN_; n++) {
                const int xcol = sxc0 + n * CW;
                const uint4 q0 = sXq[xcol].q[kk][0];
                const uint4 q1 = sXq[xcol].q[kk][1];
                dx[n] = sXd[xcol][kk];
                xq32[n][0] = (int) q0.x;
                xq32[n][1] = (int) q0.y;
                xq32[n][2] = (int) q0.z;
                xq32[n][3] = (int) q0.w;
                xq32[n][4] = (int) q1.x;
                xq32[n][5] = (int) q1.y;
                xq32[n][6] = (int) q1.z;
                xq32[n][7] = (int) q1.w;
            }
            const int row_base = ty * NROW;
            uint4 wlo = sW_lo[row_base][kk];
            uint4 whi = sW_hi[row_base][kk];
            const uint4 * dp16 = reinterpret_cast<const uint4 *>(&sWdh[kk][row_base]);

#pragma unroll
            for (int g = 0; g < NROW; g += 8) {
                const uint4 dh4 = dp16[g / 8];
                const uint16_t * dh = reinterpret_cast<const uint16_t *>(&dh4);

#pragma unroll
                for (int rr = 0; rr < 8; rr++) {
                    const int r = g + rr;

                    const uint4 wlo_c = wlo, whi_c = whi;
                    if (r + 1 < NROW) {
                        wlo = sW_lo[row_base + r + 1][kk];
                        whi = sW_hi[row_base + r + 1][kk];
                    }

                    const float d = __half2float(*reinterpret_cast<const __half *>(&dh[rr]));

#pragma unroll
                    for (int n = 0; n < TN_; n++) {
                        int idot = 0;
                        idot = ggml_cuda_dp4a((int) wlo_c.x, xq32[n][0], idot);
                        idot = ggml_cuda_dp4a((int) whi_c.x, xq32[n][4], idot);
                        idot = ggml_cuda_dp4a((int) wlo_c.y, xq32[n][1], idot);
                        idot = ggml_cuda_dp4a((int) whi_c.y, xq32[n][5], idot);
                        idot = ggml_cuda_dp4a((int) wlo_c.z, xq32[n][2], idot);
                        idot = ggml_cuda_dp4a((int) whi_c.z, xq32[n][6], idot);
                        idot = ggml_cuda_dp4a((int) wlo_c.w, xq32[n][3], idot);
                        idot = ggml_cuda_dp4a((int) whi_c.w, xq32[n][7], idot);
                        acc[r][n] += d * dx[n] * (float) idot;
                    }
                }
            }
        }
    };

    // Pipelined K loop: store current slab -> sync -> prefetch next slab -> compute -> sync.
    if (full_tile) {
        prefetch_w_full(0);
        prefetch_x_full(0, x_group);

        for (uint32_t sb0 = 0; sb0 < n_sub; sb0 += MMQ_RP_Q8_BK) {
            store_stage();
            __syncthreads();

            if (sb0 + MMQ_RP_Q8_BK < n_sub) {
                x_group += n_tok;
                prefetch_w_full(sb0 + MMQ_RP_Q8_BK);
                prefetch_x_full(sb0 + MMQ_RP_Q8_BK, x_group);
            }

            compute_stage();
            __syncthreads();
        }
    } else {
        prefetch_w_checked(0);
        prefetch_x_checked(0, x_group);

        for (uint32_t sb0 = 0; sb0 < n_sub; sb0 += MMQ_RP_Q8_BK) {
            store_stage();
            __syncthreads();

            if (sb0 + MMQ_RP_Q8_BK < n_sub) {
                x_group += n_tok;
                prefetch_w_checked(sb0 + MMQ_RP_Q8_BK);
                prefetch_x_checked(sb0 + MMQ_RP_Q8_BK, x_group);
            }

            compute_stage();
            __syncthreads();
        }
    }

    // Epilogue: write accumulators to y (float4 when the row stride is aligned;
    // MoE scatters each column through ids_dst to its destination token).
    const uint32_t rbase  = row0 + ty * NROW;
    const int      nvalid = min((int) ne1 - (int) rbase, NROW);
    const bool     vec_ok = (dst_s1 & 3) == 0;
    if (nvalid > 0) {
#pragma unroll
        for (int n = 0; n < TN_; n++) {
            const uint32_t col = tx + n * CW;
            float * dst;
            if constexpr (HAS_IDS) {
                const uint32_t a = a_base + col;
                if (!(a < a_end && col < BN)) {
                    continue;
                }
                dst = y + (size_t) ids_dst[a] * dst_s1 + rbase;
            } else {
                const uint32_t tok = tok0 + col;
                if (tok >= n_tok) {
                    continue;
                }
                dst = y + (size_t) tok * dst_s1 + rbase;
            }
            static_assert(NROW % 4 == 0, "float4 epilogue needs NROW % 4 == 0");
#pragma unroll
            for (int g = 0; g < NROW; g += 4) {
                if (vec_ok && g + 4 <= nvalid) {
                    const float4 v = { acc[g + 0][n], acc[g + 1][n], acc[g + 2][n], acc[g + 3][n] };
                    *reinterpret_cast<float4 *>(dst + g) = v;
                } else {
#pragma unroll
                    for (int r = g; r < min(g + 4, nvalid); r++) {
                        dst[r] = acc[r][n];
                    }
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, n_tok, ids_src1, ids_dst, expert_bounds, tile_off, tile_meta, n_expert, expert_stride, dst_s1);
    NO_DEVICE_CODE;
#endif
}

// Launch wrapper: 64-wide waves, tuned for large batches (ne11 >= 128).
template <bool HAS_IDS, int TN_, int NRL>
static __global__ void __launch_bounds__(64 * NRL, 2) mmq_gemm_q8_0_repacked(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t n_tok,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ tile_off,
        const repack_tile_meta * __restrict__ tile_meta,
        const uint32_t n_expert, const size_t expert_stride, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    mmq_gemm_q8_0_repacked_impl<HAS_IDS, 64, TN_, NRL>(
        wbase, xq, y, ne0, ne1, n_tok, ids_src1, ids_dst,
        expert_bounds, tile_off, tile_meta, n_expert, expert_stride, dst_s1);
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, n_tok, ids_src1, ids_dst, expert_bounds, tile_off, tile_meta, n_expert, expert_stride, dst_s1);
    NO_DEVICE_CODE;
#endif
}

// Launch wrapper: 32-wide waves, tuned for small batches.
template <bool HAS_IDS, int TN_, int NRL>
static __global__ void __launch_bounds__(32 * NRL, 3) mmq_gemm_q8_0_repacked_w32(
        const uint8_t * __restrict__ wbase, const block_q8_1 * __restrict__ xq,
        float * __restrict__ y, const uint32_t ne0, const uint32_t ne1,
        const uint32_t n_tok,
        const int32_t * __restrict__ ids_src1, const int32_t * __restrict__ ids_dst,
        const int32_t * __restrict__ expert_bounds, const int32_t * __restrict__ tile_off,
        const repack_tile_meta * __restrict__ tile_meta,
        const uint32_t n_expert, const size_t expert_stride, const uint32_t dst_s1) {
#if defined(GGML_USE_HIP) && defined(__gfx906__)
    mmq_gemm_q8_0_repacked_impl<HAS_IDS, 32, TN_, NRL>(
        wbase, xq, y, ne0, ne1, n_tok, ids_src1, ids_dst,
        expert_bounds, tile_off, tile_meta, n_expert, expert_stride, dst_s1);
#else
    GGML_UNUSED_VARS(wbase, xq, y, ne0, ne1, n_tok, ids_src1, ids_dst, expert_bounds, tile_off, tile_meta, n_expert, expert_stride, dst_s1);
    NO_DEVICE_CODE;
#endif
}
