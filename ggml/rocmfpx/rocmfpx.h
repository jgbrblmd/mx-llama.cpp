#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

#define QK_ROCMFPX 32

#define QK_ROCMFP2 QK_ROCMFPX
#define QK_ROCMFP3 QK_ROCMFPX
#define QK_ROCMFP6 QK_ROCMFPX
#define QK_ROCMFP8 QK_ROCMFPX

#define QS_ROCMFP2 ((QK_ROCMFP2 * 2) / 8)
#define QS_ROCMFP3 ((QK_ROCMFP3 * 3) / 8)
#define QS_ROCMFP6 ((QK_ROCMFP6 * 6) / 8)
#define QS_ROCMFP8 QK_ROCMFP8

#define QR_ROCMFP2 1
#define QI_ROCMFP2 (QK_ROCMFP2 / (4 * QR_ROCMFP2))

#define QR_ROCMFP3 1
#define QI_ROCMFP3 (QK_ROCMFP3 / (4 * QR_ROCMFP3))

#define QR_ROCMFP6 1
#define QI_ROCMFP6 (QK_ROCMFP6 / (4 * QR_ROCMFP6))

#define QR_ROCMFP8 1
#define QI_ROCMFP8 (QK_ROCMFP8 / (4 * QR_ROCMFP8))

// ROCmFP2/6/8 block layouts. GGUF/CPU storage uses the compact packed format.
// ROCmFP2 S40: 32 weights = 8 bytes of packed 2-bit codes + 2 UE4M3 scale
// bytes (e[0] for weights 0..15, e[1] for weights 16..31). Codes map through
// the frozen MORD code order {-4, -1, +1, +4}; value = code_value * scale.
typedef struct {
    uint8_t qs[QS_ROCMFP2];
    uint8_t e[2];
} block_rocmfp2;

// ROCmFP2 S40 code order {-4, -1, +1, +4}
static inline int rocmfpx_decode_fp2_code(uint8_t code) {
    static const int8_t values[4] = { -4, -1, 1, 4 };
    return values[code & 3u];
}

typedef struct {
    uint8_t qs[QS_ROCMFP3];
    uint8_t e[2];
} block_rocmfp3;

typedef struct {
    uint8_t qs[QS_ROCMFP6];
    uint8_t e[2];
} block_rocmfp6;

typedef struct {
    int8_t  qs[QS_ROCMFP8];
    uint8_t e;
} block_rocmfp8;

#if defined(__cplusplus)
static_assert(sizeof(block_rocmfp2) == QS_ROCMFP2 + 2*sizeof(uint8_t), "wrong rocmfp2 block size/padding");
static_assert(sizeof(block_rocmfp3) == QS_ROCMFP3 + 2*sizeof(uint8_t), "wrong rocmfp3 block size/padding");
static_assert(sizeof(block_rocmfp6) == QS_ROCMFP6 + 2*sizeof(uint8_t), "wrong rocmfp6 block size/padding");
static_assert(sizeof(block_rocmfp8) == QS_ROCMFP8 + sizeof(uint8_t), "wrong rocmfp8 block size/padding");
#else
_Static_assert(sizeof(block_rocmfp2) == QS_ROCMFP2 + 2*sizeof(uint8_t), "wrong rocmfp2 block size/padding");
_Static_assert(sizeof(block_rocmfp3) == QS_ROCMFP3 + 2*sizeof(uint8_t), "wrong rocmfp3 block size/padding");
_Static_assert(sizeof(block_rocmfp6) == QS_ROCMFP6 + 2*sizeof(uint8_t), "wrong rocmfp6 block size/padding");
_Static_assert(sizeof(block_rocmfp8) == QS_ROCMFP8 + sizeof(uint8_t), "wrong rocmfp8 block size/padding");
#endif

GGML_API float  rocmfpx_ue4m3_to_fp32(uint8_t e);
GGML_API bool   rocmfpx_scale_is_valid(uint8_t e);
GGML_API size_t rocmfpx_row_size_fp2(int64_t k);
GGML_API size_t rocmfpx_row_size_fp3(int64_t k);
GGML_API size_t rocmfpx_row_size_fp6(int64_t k);
GGML_API size_t rocmfpx_row_size_fp8(int64_t k);

GGML_API void   rocmfpx_quantize_row_fp2_ref(const float * GGML_RESTRICT x, block_rocmfp2 * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_dequantize_row_fp2(const block_rocmfp2 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_quantize_row_fp2(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t rocmfpx_quantize_fp2(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
// affine fp2 variant (value = c*scale - offset, e[0] = scale, e[1] = offset)
GGML_API void   rocmfpx_quantize_row_fp2_affine_ref(const float * GGML_RESTRICT x, block_rocmfp2 * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_dequantize_row_fp2_affine(const block_rocmfp2 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_quantize_row_fp2_affine(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t rocmfpx_quantize_fp2_affine(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);
// detect the fp2 variant from block data: true = affine, false = S40 codebook
GGML_API bool   rocmfpx_fp2_is_affine(const uint8_t * data, size_t nbytes);

GGML_API void   rocmfpx_quantize_row_fp3_ref(const float * GGML_RESTRICT x, block_rocmfp3 * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_dequantize_row_fp3(const block_rocmfp3 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_quantize_row_fp3(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t rocmfpx_quantize_fp3(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API void   rocmfpx_quantize_row_fp6_ref(const float * GGML_RESTRICT x, block_rocmfp6 * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_dequantize_row_fp6(const block_rocmfp6 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_quantize_row_fp6(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t rocmfpx_quantize_fp6(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

GGML_API void   rocmfpx_quantize_row_fp8_ref(const float * GGML_RESTRICT x, block_rocmfp8 * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_dequantize_row_fp8(const block_rocmfp8 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k);
GGML_API void   rocmfpx_quantize_row_fp8(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k);
GGML_API size_t rocmfpx_quantize_fp8(const float * GGML_RESTRICT src, void * GGML_RESTRICT dst, int64_t nrows, int64_t n_per_row, const float * imatrix);

#ifdef __cplusplus
}
#endif
