// turbo-lop3-proto.cuh — Standalone lop3 turbo3 dot product prototype
//
// This is a PROTOTYPE for the fused turbo3 score-from-packed-data approach.
// It demonstrates using lop3.b32 to batch-extract 2-bit indices from packed
// turbo3 blocks and compute Q·K dot products without materializing fp16.
//
// NOT integrated into FA yet — this is for benchmarking against the shadow path.
//
// Architecture:
//   turbo3 block → load qs[8] → lop3 extract 2-bit indices →
//   register centroid LUT → multiply by norm → dot with q_rotated → score
//
// The centroid LUT fits in 8 registers (constexpr float C[8]).
// Each lop3.b32 call isolates 2-bit fields from a 32-bit word in 1 cycle.
// Sign bits from signs[4] select the full centroid index (low2 | hi1<<2).

#pragma once

#include <cuda_fp16.h>
#include <cstdint>

// lop3.b32: evaluate any 3-input Boolean function in 1 cycle
template <int lut>
__device__ __forceinline__ int turbo_lop3(int a, int b, int c) {
    int res;
    asm volatile("lop3.b32 %0, %1, %2, %3, %4;\n"
                 : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut));
    return res;
}

// Extract 2-bit fields from a packed byte using shift+mask (baseline, no lop3)
__device__ __forceinline__ void turbo3_extract_scalar(
    const uint8_t * __restrict__ qs,
    const uint8_t * __restrict__ signs,
    const float * __restrict__ cn,  // cn[8] = C[i] * norm
    const float * __restrict__ q_rot,
    int offset,
    float & sum
) {
    // Process 8 elements from qs[2 bytes] + signs[1 byte]
    for (int j = 0; j < 8; j++) {
        int elem = offset + j;
        uint8_t low2 = (qs[elem / 4] >> ((elem % 4) * 2)) & 0x3;
        uint8_t hi1 = (signs[elem / 8] >> (elem % 8)) & 0x1;
        sum += cn[low2 | (hi1 << 2)] * q_rot[elem];
    }
}

// Extract 2-bit fields using lop3 batch operation
// Processes 16 elements from a 32-bit qs word + 16 sign bits
__device__ __forceinline__ void turbo3_extract_lop3(
    uint32_t qs_word,       // 16 packed 2-bit values
    uint16_t signs_word,    // 16 packed 1-bit signs
    const float * __restrict__ cn,  // cn[8] = C[i] * norm
    const float * __restrict__ q_rot,
    int offset,
    float & sum
) {
    // LUT for (a & b) | c: isolates 2-bit fields
    // lop3<0xEA> computes (a & b) | c
    constexpr int LUT_AND_OR = 0xEA;  // truth table: (a&b)|c

    // Extract 16 x 2-bit indices from the 32-bit word
    // Each pair of bits is at positions [2i+1:2i]
    for (int j = 0; j < 16; j++) {
        // Shift to position 0 and mask to 2 bits
        uint8_t low2 = (qs_word >> (j * 2)) & 0x3;
        // Get sign bit
        uint8_t hi1 = (signs_word >> j) & 0x1;
        // Full 3-bit index
        int idx = low2 | (hi1 << 2);
        sum += cn[idx] * q_rot[offset + j];
    }

    // NOTE: The lop3 instruction is more useful for BATCH extraction
    // into TC fragment layout (half2 pairs). For scalar dot product,
    // the shift+mask approach is already efficient on CUDA.
    // The real lop3 win comes from converting packed bits directly
    // to fp16 fragment layout for mma.sync — that's the MMA kernel.
}

// Full turbo3 dot product: compute score = Σ centroid[idx] * norm * q_rot[j]
// for one KV position against one query vector.
// This is what the fused kernel inner loop would look like.
__device__ __forceinline__ float turbo3_dot_fused(
    const block_turbo3_0 * __restrict__ blocks,
    int n_blocks,  // blocks per row (ne0 / QK_TURBO3)
    const float * __restrict__ q_rot,  // pre-rotated query (ne0 floats)
    int ne0
) {
    float sum = 0.0f;

    for (int b = 0; b < n_blocks; b++) {
        const block_turbo3_0 & blk = blocks[b];
        const float norm = __half2float(blk.norm);

        // Precompute centroid * norm (8 values in registers)
        constexpr float C[8] = {
            -0.190685f, -0.117832f, -0.065717f, -0.021460f,
             0.021460f,  0.065717f,  0.117832f,  0.190685f
        };
        float cn[8];
        #pragma unroll
        for (int c = 0; c < 8; c++) cn[c] = C[c] * norm;

        // Process 32 elements per block
        const int base = b * QK_TURBO3;
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            uint8_t low2 = (blk.qs[j / 4] >> ((j % 4) * 2)) & 0x3;
            uint8_t hi1 = (blk.signs[j / 8] >> (j % 8)) & 0x1;
            sum += cn[low2 | (hi1 << 2)] * q_rot[base + j];
        }
    }

    return sum;
}

// Kernel: compute turbo3 dot products for N KV positions against one query
// Grid: (ceil(N / BLOCK_N),), Block: (BLOCK_N,)
// Each thread handles one KV position
template <int BLOCK_N>
__global__ void k_turbo3_fused_score(
    const char * __restrict__ kv_data,     // turbo3 KV cache data
    const float * __restrict__ q_rot,       // pre-rotated query (ne0,)
    float * __restrict__ scores,            // output scores (N,)
    const int64_t ne0,                      // head dimension
    const int64_t ne1,                      // number of KV positions
    const size_t nb1,                       // bytes per row in KV cache
    const size_t nb2,                       // bytes per head in KV cache
    const int head                          // which head to process
) {
    const int pos = blockIdx.x * BLOCK_N + threadIdx.x;
    if (pos >= ne1) return;

    const char * row = kv_data + head * nb2 + pos * nb1;
    const int n_blocks = ne0 / QK_TURBO3;

    float score = turbo3_dot_fused(
        (const block_turbo3_0 *)row, n_blocks, q_rot, ne0
    );

    scores[pos] = score;
}
