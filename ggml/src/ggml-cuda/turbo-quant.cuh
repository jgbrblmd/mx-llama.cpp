/*
 * turbo-quant.cuh — TurboQuant turbo3 CUDA kernel declarations
 *
 * Part of the turbo3 CUDA port for llama.cpp.
 * Declares the TURBO_WHT op and SET_ROWS turbo3 quantize dispatch.
 *
 * Author: Erol Germain (@erolgermain)
 * Date:   March 2026
 * License: MIT (matching upstream llama.cpp)
 */

#pragma once

#include "common.cuh"

void ggml_cuda_op_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_set_rows_turbo2(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_set_rows_turbo3(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_set_rows_turbo4(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
