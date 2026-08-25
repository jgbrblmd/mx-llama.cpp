# jgbrblmd 移植功能文档

**创建日期：** 2026-08-25  
**参考分支：** `jgbrblmd-sync`（基于 mxxm-t/master，已合并 fork/master 47 commits）  
**上游基准：** mxxm-t/master（当前 def64b335）

本文档按功能模块分类记录从 `fork/master` 移植到 `jgbrblmd-sync` 的所有特性。
每次重新 rebase 前可对照本文检查是否遗漏重要功能。

---

## 目录

1. [ROCmFPX / ROCmFP4 / ROCmFP6 / ROCmFP8 量化类型](#1-rocmfpx--rocmfp4--rocmfp6--rocmfp8-量化类型)
2. [Lucebox DSV4-Flash MIX-STRIX per-expert mix 量化](#2-lucebox-dsv4-flash-mix-strix-per-expert-mix-量化)
3. [gfx906 GPU 内核优化](#3-gfx906-gpu-内核优化)
4. [模型类型支持](#4-模型类型支持)
5. [投机解码（Speculative Decoding）](#5-投机解码speculative-decoding)
6. [GDN / Delta-net / Qwen MoE 修复](#6-gdn--delta-net--qwen-moe-修复)
7. [量化转换器工具](#7-量化转换器工具)
8. [服务端改进与 Bug 修复](#8-服务端改进与-bug-修复)
9. [推理引擎核心修复](#9-推理引擎核心修复)
10. [已遗漏 / 待移植（仍在 fork/master）](#10-已遗漏--待移植仍在-forkmaster)

---

## 1. ROCmFPX / ROCmFP4 / ROCmFP6 / ROCmFP8 量化类型

### 1.1 基础类型注册与 CPU/GPU 内核

| Commit | 说明 |
|--------|------|
| `ce309996d` | 首次移植：添加 `GGML_TYPE_Q4_0_ROCMFP4`/`Q6_0_ROCMFP6`/`Q8_0_ROCMFP8`（枚举 100–103），含 block layout、CPU quantize/dequantize、vec_dot、ftype 映射。gfx906 GPU MMQ/MMVQ kernel（dp4a tile，512 threads）。修复 FP6/FP8 load_tiles sram_stride 与 vec_dot stride 不匹配问题 |
| `5edbe83a8` | 添加 MXFP4 全模型量化 + ROCmFP4/FP6/FP8 类型验证；gguf-py type id 对齐 fork C enum（MXFP4=39, NVFP4=40） |
| `f8c70c5e5` | 添加 `Q2_0_ROCMFPX`（affine fp2，即 Q2_0 + affine scale），gfx906 kernel，ftype 112 |

### 1.2 NVFP4_E8M0 / ROCmFP4 STRIX

| Commit | 说明 |
|--------|------|
| `60f6a1b9a` | 添加 `NVFP4_E8M0`（ModelOpt 格式，log2-fixed-point scale）；laguna MoE per-expert scale fix（C=9300 常数重缩放）；gfx906 NVFP4 mmq config；streaming safetensors→NVFP4 GGUF converter + torch quantizer |
| `4d1f0e94d` | laguna 工具链从 NVFP4 全面切换到 ROCmFP4（精度/速度均更优）；保留 NVFP4 仅用于文件兼容 |
| `5e4fdf863` | 添加 ROCmFP4 STRIX 支持：`--rocmfp4` 写 STRIX recipe（embd Q6_K, attn k/v Q4_0_ROCMFP4, rest FAST, norms F32），ftype 105 `MOSTLY_Q4_0_ROCMFP4_STRIX`；新增 `convert-strix-lean-rocmfp4.py`（bf16→STRIX_LEAN ftype 106）；MLX mxfp4 支持：读取 quantization_config.mode，dequantize e2m1 codes + E8M0 scales；Qwen3Next 修复：文件级 RMSNorm +1 offset 检测（跳过 unsloth GAIN 1-centered norm） |

### 1.3 Q2_0_ROCMFPX 拆分

| Commit | 说明 |
|--------|------|
| `41ecfef67` | 将 Q2_0_ROCMFPX（原 ftype 112）拆分为两个独立类型：S40 codebook（qtype 107）和 affine fp2（qtype 108），避开枚举冲突，支持两种不同代码本 |

### 1.4 Q3_0_ROCMFPX (ROCmFP3)

| Commit | 说明 |
|--------|------|
| `c64d4669f` | 添加 `Q3_0_ROCMFPX`（即 ROCmFP3）类型，含 MMQ/MMVQ kernels；ftype 114（避开 112 冲突） |

---

## 2. Lucebox DSV4-Flash MIX-STRIX per-expert mix 量化

| Commit | 说明 |
|--------|------|
| `342e90367` | **核心移植**：添加 qtype 105（Q3_1_ROCMFP3_MIX）/ 106（Q2_1_ROCMFP2_MIX），per-expert codebook，GGUF KV sidecar（`deepseek4.gumix.sidecar` / `deepseek4.p4mix.sidecar`）。sync-free mul_mat_id decode hook（ne12==1），supports_op 白名单，graph-usability guard。loader 解析嵌入 sidecars，按设备切片注册 mix tensor |
| `b9a79a3ec` | gfx906 fuse mix-qtype gate/up + clamp + SwiGLU 到单个 decode launch，减少 kernel 启动开销 |
| `ced43853f` | 添加调试开关 `GGML_CUDA_DISABLE_MIX_MATVEC`，用于隔离 mix-matvec 相关问题 |

---

## 3. gfx906 GPU 内核优化

### 3.1 NVFP4 / Q5_K MMQ

| Commit | 说明 |
|--------|------|
| `330ee780b` | NVFP4 k-loop unroll：应用与 Q8_0 相同的 k-loop unroll 策略到 NVFP4，减少地址计算；gfx906 (VEGA20) 用 dp4a 而非 INT8 MFMA；预期 prefill +10–15% |
| `53bc07ce5` | gfx906 Q5_K MMQ 8-warp wide-tile config，提升 matmul 吞吐量 |

### 3.2 GDN 内核修复（gfx906 双 bug）

| Commit | 说明 |
|--------|------|
| `d042478a3` | **修复 DPP XOR8 reduce bug**：chunked GDN kernel 中 DPP XOR8 reduce 指令错误导致结果错误（已验证 7e90f0229 根因） |
| `c7b085040` | **修复 interleaved 头配对缺失**：chunked GDN kernel 缺少 interleaved head pairing，导致多头注意力计算错误 |
| `2aa2ecf10` | 修复 fused GDN v/k head pairing 为 grouped order（与 MLX 约定对齐） |

### 3.3 Q/GDN q/k head expansion

| Commit | 说明 |
|--------|------|
| `6cd41cefd` | qwen35moe GDN q/k head expansion 修正为 grouped order |
| `2e20d9c75` | qwen35moe default GDN v-head pairing 改为 tiled（修复 BigBang-v1 乱码根因） |

---

## 4. 模型类型支持

### 4.1 Muse-glimmer

| Commit | 说明 |
|--------|------|
| `9962db618` | 添加 muse-glimmer 模型类型：converter（`conversion/muse_glimmer.py`）、chat template、GGUF 常量注册、arch 识别、model loader；179 行 converter，208 行 model 实现 |

### 4.2 Bailingmoe3

| Commit | 说明 |
|--------|------|
| `7851ac458` | **初始移植**：添加 bailingmoe3 模型类型支持（139 行 converter），注册 GGUF 常量和 tensor mapping，arch 识别 |
| `e3b4a6387` | **修复**：使 `kda_safe_gate` 可选（兼容 AtomicBot 导出器），修复 ssm_f/ssm_g 张量名不匹配导致加载失败的问题 |
| `641b4411b` | **Bug fix**：Bailing tool-call parser 中 value_suffix 误带换行，导致参数混入 XML 标签；已修复验证（109 测试通过） |

### 4.3 Delta-net / GDN CPU 推理

| Commit | 说明 |
|--------|------|
| `17a65ea7d` | delta-net chunked GDN decay_mask 方向修复 |
| `8a856e08d` | qwen* GDN CPU 推理：使 beta/gate 在 reshape_4d 后 contiguous |

### 4.4 Qwen3.5 / Qwen3Next 修复

| Commit | 说明 |
|--------|------|
| `5e06a5b2d` | 修复 Qwen3.5 norm +1 bake 问题；写入 `ssm.v_heads_tiled` 配对标志到 GGUF KV |
| `d11533991` | MLX affine quantized weights 支持（u32 dequant）+ norm-offset flag |

### 4.5 GDN v-head pairing 约定

| Commit | 说明 |
|--------|------|
| `042aca07b` | GDN v-head pairing 改为 per-checkpoint 配置（tiled vs grouped），通过 GGUF KV override 控制 |
| `2e20d9c75` | qwen35moe default 改为 tiled（BigBang-v1 乱码根因修复）|

---

## 5. 投机解码（Speculative Decoding）

### 5.1 DFlash / DFlash2

| Commit | 说明 |
|--------|------|
| `186d1fbc7` | **初始移植**：添加基础投机解码支持 |
| `0a7c453b9` | 支持 DFlash2（draft model）加载和推理 |
| `594ab0927` | 修复 DFlash2 `n_outputs_max` + graph node count bug |
| `5a0ded4a4` | **完整 DFlash2 移植**（PR #27342）：d2t draft-to-target vocab mapping、fc_s feature fusion scale tensor、ffn_up_s/ffn_gate_s/ffn_down_s scale tensors（量化模型）、sample_from_anchor / bonus anchor slot、p_min 概率阈值、TENSOR_ALLOW_RESHAPE flag、dsv4_o_group_count assert |
| `e0436eaa3` | 启用 DFlash draft models 的 tensor mirror（减少跨设备传输） |
| `46015720c` | DFlash2 完整支持（与 5a0ded4a4 互补，含额外修复） |

### 5.2 投机解码配置重构

| Commit | 说明 |
|--------|------|
| `2d7d41d1e` | 重构 `common_speculative_init`：将 spec decode 启用配置统一到 common 层 |
| `545ea4eb9` | **自动检测 spec type**：从 draft GGUF metadata 自动推断 spec type（无需手动指定 `--spec-type`）|
| `2ecf5b799` | 将 spec-decode 计数器添加到 `/metrics` 端点，便于监控 |

---

## 6. GDN / Delta-net / Qwen MoE 修复

### 6.1 GDN v/k head pairing（grouped 顺序）

| Commit | 说明 |
|--------|------|
| `6cd41cefd` | qwen35moe GDN q/k head expansion 修正为 grouped order（与 MLX 约定对齐）|
| `2aa2ecf10` | fused GDN v/k head pairing 修正为 grouped order |

### 6.2 KQ mask 重计算

| Commit | 说明 |
|--------|------|
| `07b96f959` | llama-kv-cache：每 token 重新计算 KQ mask，统一 KV cache 行为；解决 chunked attention 中 mask 不一致问题 |

---

## 7. 量化转换器工具

### 7.1 Laguda / STRIX / Jormungandr

| Commit | 说明 |
|--------|------|
| `3fd67c84d` | laguna stream conversion：`make-laguna-shell.py` 生成纯 metadata GGUF shell（跳过 230GB BF16 中间文件），`convert-laguna-stream.py` 写入 ROCmFP4 tensor 数据；HF safetensors row-major == GGUF numpy order，无需 transpose |
| `7723cb6df` | Jormungandr ROCmFP4 转换器：`tools/convert-jormungandr-rocmfp4.py`，修复 3 个隐性 bug（14.99 GB 输出验证通过）|

### 7.2 STRIX / MLX

| Commit | 说明 |
|--------|------|
| `5e4fdf863` | STRIX converter：`convert-strix-lean-rocmfp4.py`（bf16→STRIX_LEAN ftype 106）；MLX mxfp4 support in base.py |

---

## 8. 服务端改进与 Bug 修复

| Commit | 说明 |
|--------|------|
| `42e52f566` | 添加 `--prefill-priority` phased batch scheduling：按 prefill 优先级调度请求，提升吞吐 |
| `1ccbcc331` | 修复 prefill_priority merge resolution 在 pre_decode 中损坏的问题 |
| `a95102a20` | 修复 OOM 时 reserve 失败被忽略导致 NULL vbuffer segfault；修复 null-task assert |
| `27635a3c9` | ggml-backend-meta：OOM 时 meta buffer allocation assert 改为 graceful fallback |

---

## 9. 推理引擎核心修复

| Commit | 说明 |
|--------|------|
| `c5c288b92` | llama：attn output weight 按 GQA-group 粒度切分（而非逐 head），修复 4 卡 `-sm tensor` 时 wo 层切分与 Q 侧 GQA 组粒度不对齐导致的崩溃 |

---

## 10. 已遗漏 / 待移植（仍在 fork/master）

以下 commits 目前在 `fork/master` 上，尚未 port 到 `jgbrblmd-sync`：

| Commit | 说明 |
|--------|------|
| `330ee780b` | gfx906 NVFP4 k-loop unroll prefill 优化（已在 jgbrblmd-sync 中，见 3.1）|
| `46015720c` | DFlash2 完整支持（已在 jgbrblmd-sync 中，见 5.1）|

> **注意：** 上述两个 commit hash 仅在 fork/master 上，但功能内容已包含在 jgbrblmd-sync 的 port 历史中（通过 merge commit `a76d48d01` 带入）。实际尚未 port 的是以下 3 个上游 PR（已在 mxxm-t/master 之外）：
> - **#26389** — 投机解码计数器到 `/metrics` 端点
> - **#26510** — 投机解码配置重构（`common_speculative_init`）
> - **#26814** — 从 draft GGUF metadata 自动检测 spec type

### fork/master 独有但尚未 port 的 commits

| Commit | 说明 |
|--------|------|
| `5e4fdf863` (fork) | ROCmFP4 STRIX + Qwen3Next MLX — **已在 jgbrblmd-sync**（见 1.2）|
| `a95102a20` | fix null vbuffer segfault after failed reserve — **已在 jgbrblmd-sync**（见 8）|
| ... | （其余 47 commits 全部已 port，详见上文各节）|

**当前 fork/master 比 jgbrblmd-sync 多出的内容：**
- 仅包含 commit `330ee780b` (gfx906 NVFP4 k-loop unroll) 的独立版本（jgbrblmd-sync 中有等价功能但 hash 不同）

**真正未 port 的只有 mxxm-t/master 侧的新 PR：**
- #26389, #26510, #26814（投机解码相关，等待上游合并后再 rebase）

---

## Rebase 检查清单

每次从 `mxxm-t/master` rebase 时，请按以下步骤检查：

1. **编译验证**：`cmake --build . --target llama-server -j` 确认无错误
2. **量化类型枚举**：确认 qtype 105/106/107/108 未与上游冲突
3. **DFlash2 功能**：确认 d2t mapping、fc_s scale tensor、anchor slot 正常工作
4. **GDN pairing**：确认 tiled/grouped 约定正确（BigBang-v1 需 tiled）
5. **Bailing tool-call parser**：确认 value_suffix 无换行尾缀
6. **投机解码计数器**：#26389/#26510/#26814 合并到 mxxm-t/master 后重新 port

---

## Git 拓扑速查

```
mxxm-t/master (def64b335) ──────────── a76d48d01 (merge) ── 87849eb39 ── 6d5c12ac0 (jgbrblmd-sync HEAD)
                                            ↑ fork/master (ef03a6173) 的 47 commits 通过 merge 带入
fork/master (330ee780b) ──────────────────────────────────────────────────── (独立演进，2 commits 领先)
```
