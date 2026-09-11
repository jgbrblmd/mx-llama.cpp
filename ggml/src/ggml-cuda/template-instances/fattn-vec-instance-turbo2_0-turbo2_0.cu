/*
 * fattn-vec-instance-turbo2_0-turbo2_0.cu — Flash Attention template instances
 *
 * Instantiates FA vec kernels for turbo2 K + turbo2 V at head dims 64/128/256.
 *
 * Author: Erol Germain (@erolgermain)
 * Date:   March 2026
 * License: MIT (matching upstream llama.cpp)
 */

#include "../fattn-vec.cuh"

DECL_FATTN_VEC_CASE( 64, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0);
DECL_FATTN_VEC_CASE(128, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0);
DECL_FATTN_VEC_CASE(256, GGML_TYPE_TURBO2_0, GGML_TYPE_TURBO2_0);
