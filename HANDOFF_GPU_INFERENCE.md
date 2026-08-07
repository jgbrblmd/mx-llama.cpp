# 交接文档：KAT-Coder (Qwen3.5MoE) 推理修复（2026-08-07 更新）

> 会话日期：2026-08-07
> 目标：让 `mx-llama.cpp` fork 在 gfx906 (ROCm 7.2.4) 上正确推理 `/opt/LLM/hf/coder` 模型（✅ 已达成：CPU 与 GPU 均正确）

---

## 1. 环境与模型（不变）

| 项目 | 值 |
|---|---|
| 模型目录 | `/opt/LLM/hf/coder`（MLX affine 量化 8bit/6bit，40 层 = 30 linear-attn + 10 full-attn） |
| bf16 GGUF | `/opt/LLM/hf/coder-bf16.gguf`（69GB，**有效**；另有调查期间生成的 `coder-bf16-v2.gguf` 内容相同，可删） |
| 量化 GGUF | `/opt/LLM/hf/coder-rocmfp4.gguf`（19.7GB） |
| GPU | 5× gfx906 34GB，ROCm 7.2.4，构建 `build-hip/` |
| **MLX 参考** | `/opt/venv/common/bin/python` + mlx 0.32 + mlx-lm 0.31.3（CPU-only） |

## 2. ✅ CPU 推理已修复并验证（commit 7c7d831d7）

**根因：q/k 头扩展用了 tiled 序，模型需要 grouped 序。**

- `src/models/qwen35moe.cpp` `build_layer_attn_linear`：原来 `ggml_repeat_4d`（tiled
  [k0..k15, k0..k15]）→ 改为按 k-head 切片 concat r=2 次（grouped [k0,k0,k1,k1,...]）。
  **v-head h 必须配对 k-head h//r**（MLX 用 `mx.repeat`；层 0 GDN 输出对拍：grouped vs
  MLX 差 0.025，tiled 差 0.334）。
- 上一会话的 decay_mask `ggml_neg`（commit 8d03248b9）是**误修**，已还原：原版
  `sub+LOWER_DIAG` 与 CPU solve_tri 自洽，保留区 exp(gc[j]-gc[i])≤1 是正确衰减；
  neg 后变增长 → inf/NaN（len≥2 的 logprobs 全 null 就是它）。

**端到端验证（CPU，bf16，temp=0）：**

| prompt | CPU bf16 | MLX 8-bit |
|---|---|---|
| Hello | `,` (-0.82) | `,` ✓（存档 top5 同族） |
| Hello world | `!` | `!` ✓ |
| Hello world! | `\n\n` (271) | `\n\n` (271) ✓ |
| Write a Python hello world | ` program` (1957) | `\n\n` (271)（**同一 top-2 互换**，bf16 vs 8bit 正常差异） |
| 40-token 生成 | 流畅输出 `print("Hello, World!")` 代码 | — |

**测试方法**（继续用）：`llama-server -ngl 0 -t 24` 后台 + curl /health 等端口 +
/completion（n_predict=1, temperature=0, top_k=1, n_probs）→ 分析 → `pkill -x llama-server`。
**不要用 llama-cli**（交互退不出）。参考脚本：`/tmp/cpu_repro.sh`。

## 3. 调查过程中的伪影教训（别再踩）

1. **gguf-python 读 token_embd**：文件是 (248320, 2048) 行主序，按 (2048, 248320)
   reshape 会读出错位垃圾（当时差点误判"GGUF 表错了"，甚至重转了 69GB 的 v2）。
   正确：`np.frombuffer(...).view(bfloat16).reshape(248320, 2048)`。
2. **server 加载时 fused-op probe 图**：模型加载会跑 2 个 probe 图（token
   [BOS,EOS] 和 [0,0]），真正的 prefill 是第 3 个图（图 02+）。dump 时注意图序号
   （layer_00 的 map 回调自增计数，layer≥1 的图号 +1）。
3. **带步长视图的 dump**：对 q/k/v 这类 strided view，`memcpy(ggml_nbytes)` 只拷逻辑
   跨度的一部分，会得到"t=1 的 v == Q 区"的假象；要按 nb 逐元素拷贝。
4. **flat 布局**：ggml ne0 连续 = numpy (T,H,S) 同序；视图/permute 后按 nb 推算；
   decay_mask 的 tri 在 exp 前（掩码区 = exp(0) = 1.0，不是 0）。

## 4. ✅ GPU fused GDN 已修复（commit 1382b8e83，2026-08-07）

**根因与 CPU 完全同源**：fused 内核 `ggml/src/ggml-cuda/gated_delta_net.cu` 用
`fastmodulo(h_idx, neqk1)`（tiled 序）取 k-head，而模型要求 grouped 序
（v-head h 配对 k-head h//r，r = H_v/H_k = 32/16 = 2）。修复：`fastdiv(h_idx, r)`
（r magic 在 launch 处 `init_fastdiv_values(H / neqk1)` 计算）+ 断言 `nev1 % neq1 == 0`。
r=1 时两种索引相同，其他模型不受影响。

**GPU 验证（单卡 gfx906，coder-rocmfp4.gguf 19.7GB，-ngl 99，temp=0 top_k=1）：**

| prompt | GPU 8-bit fused | CPU bf16 | MLX 8-bit |
|---|---|---|---|
| Hello | `,` ✓ | `,` ✓ | `,` ✓ |
| Hello world | `!` ✓ | `!` ✓ | `!` ✓ |
| Hello world! | `\n\n` (271) ✓ | `\n\n` (271) ✓ | `\n\n` (271) ✓ |
| Write a Python hello world | ` program` (1957) ✓ | ` program` (1957) ✓ | `\n\n`（top-2） |

40-token 生成与 CPU bf16 同 prompt 逐字对比：仅首 token 互换了 bf16/8bit 的正常
top-2（`.\n\n` vs `\n\n`），其余完全一致（`<think>\nHere's a thinking process...`）。
GPU 70 tok/s decode / 60 tok/s prefill。

**测试脚本**：`/tmp/gpu_repro.sh <port>`（top-1 复现）、`/tmp/gpu_gen40.sh <port>`（40-token）。
注意机器上常驻 prod server（Qwen3.6-27B，4 卡，port 1302，占卡 0/1/2/4）——测试用卡 3，
`HIP_VISIBLE_DEVICES=3`。

## 5. 剩余问题（按优先级）

1. **GPU chunked 路径 rocblas 崩溃（原 4.3）**：`ggml_solve_tri` → `hipblasStrsmBatched` →
   ROCm 7.2.4 库 bug。CPU 已验证 chunked 数学正确；GPU 上 fused 已可用，**绕开即可**，
   无需修库（`--no-fused-gdn-ch` 才需要）。
2. **bf16 vs 8-bit top-1 可互换（原 4.4）**：正常现象，非 bug。

## 6. 工具资产（/tmp/，可复用）

- `cpu_repro.sh` — 4 长度 prompt top-1 复现（llama-server + curl）
- `gpu_repro.sh <port>` — 同上，GPU 单卡（默认卡 3，脚本内 HIP_VISIBLE_DEVICES 可改）
- `gpu_gen40.sh <port>` — 40-token 生成 + 计时（GPU）
- `full_groundtruth.py` — numpy 全 40 层模型（**q/k repeat 用 np.repeat=grouped 是对的**）
- `llama_chunk_variants.py` — ⚠️ conv 用 `reshape(4,8192).T` 是**错的**（应 `reshape(8192,4)`），仅作历史参考
- `cmp_gdn_full.py` / `cmp_l0_components.py` — 层 0/GDN 逐张量对拍（注意 flat 布局）
- `mlx_l0_gdn.py` — MLX 层 0 GDN 逐头输出（gated_delta_ops + grouped repeat）
- `/tmp/mlx_emb2.npy` = "Hello world" 的 token embedding（MLX 生成，正确）
- `/tmp/mlx_logprobs.npy` = 5-token prompt 的 MLX logprobs（argmax 271）；`mlx_logprobs_hello.npy` = "Hello"（argmax 11）

## 7. 关键代码位置

- `src/models/qwen35moe.cpp:460-490` — q/k grouped 扩展（CPU 修复处，commit 7c7d831d7）
- `src/models/delta-net-base.cpp:102-144` — decay_mask（sub+LOWER_DIAG，勿加 neg）
- `ggml/src/ggml-cuda/gated_delta_net.cu:37-42` — fused 内核 k-head 索引 `fastdiv(h_idx, rq1_magic)`（GPU 修复处，commit 1382b8e83）；launch 处 r magic = `init_fastdiv_values(H / neqk1)`
- `src/llama-context.cpp:250-251` — fused_gdn_ar/ch 默认 true
