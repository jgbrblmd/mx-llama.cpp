// Standalone test: full CT_INT4 matmul path (F32 activation -> Q8_0 + CT_INT4 dot).
// Uses a hand-computable pattern so the expected output is exact.
#include "ggml.h"
#include "ggml-cpu.h"
#include <stdio.h>
#include <math.h>
#include <vector>
#include <cstring>

int main() {
    const int64_t K = 128;   // one CT_INT4 block
    const int64_t N = 2;     // output rows
    const int64_t row_bytes = (K/128)*68;  // 68

    // Build CT_INT4 weight with known dequantized values:
    //   row 0: all weights = 1.0  (int = 9  (9-8=1), scale = 1.0)
    //   row 1: all weights = 2.0  (int = 10 (10-8=2), scale = 1.0)
    std::vector<char> wdata(N*row_bytes, 0);
    for (int64_t i = 0; i < N; i++) {
        char* row = wdata.data() + i*row_bytes;
        uint16_t one_fp16 = 0x3C00;   // 1.0 in fp16
        memcpy(row, &one_fp16, 2);    // d at offset 0
        // pad (offset 2, 2 bytes) already zero
        uint32_t qval   = (i == 0 ? 9 : 10);
        uint32_t packed = 0;
        for (int s = 0; s < 8; s++) packed |= (uint32_t)(qval << (s*4));
        uint32_t* qs = (uint32_t*)(row+4);
        for (int j = 0; j < 16; j++) qs[j] = packed;  // all 16 int32 = packed
    }

    // Activation x: K values, all = 1.0 (Q8_0-quantizes 1.0 exactly)
    std::vector<float> xdata(K, 1.0f);

    struct ggml_init_params params = { .mem_size = 16*1024*1024, .mem_buffer = nullptr };
    struct ggml_context * ctx = ggml_init(params);
    int64_t dw[3] = {K, N, 1};
    struct ggml_tensor * w = ggml_new_tensor(ctx, GGML_TYPE_CT_INT4, 2, dw);
    memcpy(w->data, wdata.data(), wdata.size());
    int64_t dx[3] = {K, 1, 1};
    struct ggml_tensor * x = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, dx);
    memcpy(x->data, xdata.data(), K * sizeof(float));

    struct ggml_tensor * out = ggml_mul_mat(ctx, w, x);   // [N, 1]

    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    enum ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, 1);   // single thread

    float* o = (float*)out->data;
    printf("compute status=%d\n", (int)st);
    printf("out[0] = %f  (expected 128.0)\n", o[0]);
    printf("out[1] = %f  (expected 256.0)\n", o[1]);
    printf("diff0 = %f, diff1 = %f\n", fabsf(o[0]-128.0f), fabsf(o[1]-256.0f));
    return 0;
}
