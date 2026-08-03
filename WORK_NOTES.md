# 工作交接：mx-llama.cpp gfx906 NVFP4 优化 + Laguna 模型转换

## 环境
- 代码库：/opt/LLM/working/mx-llama.cpp（ROCm fork，HIP backend，主分支 master）
- 硬件：5× gfx906 (MI50) 各 32GB，HIP_VISIBLE_DEVICES 控制
- Python：/opt/venv/common/bin/python（torch 2.11 CPU + safetensors）
- 构建目录：build-hip（`-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx906 -DGGML_CUDA_FA=OFF -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_APP=OFF`，ccache 已启用）
- 二进制在 build-hip/bin（llama-cli/llama-bench/llama-quantize/test-backend-ops），运行时需 `LD_LIBRARY_PATH=build-hip/bin`
- 模型：/opt/LLM/Qwen3.5-0.8B-BF16.gguf（1.4GB，验证用）、/opt/LLM/Qwen3.5-0.8B-NVFP4.gguf（自量化验证用）
- 网络：HF 需 HK 代理 `-x http://127.0.0.1:6117`

## 任务一：gfx906 上 NVFP4 优化【已完成并验证】
改动 4 个文件（未提交，工作树状态）：
1. `ggml/src/ggml-cuda/mmq-config-gfx906.cuh` — NVFP4 从 rdna2 通用配置（256 线程/J≤64）改为 gfx906 专用：512 线程 + J≤128（occupancy gate 保护小形状）
2. `ggml/src/ggml-cuda/common.cuh` — `ggml_cuda_ue4m3_to_fp32` 软件路径改为纯位操作（`((exp+119)<<23)|(man<<20)`，次正规 `man*2^-10`），256 输入穷举 bit-exact
3. `tools/quantize/quantize.cpp` — ftype 表补 `NVFP4` 条目（原缺失）
4. `src/llama-quant.cpp` — `llama_ftype_get_default_type` 补 `MOSTLY_NVFP4→GGML_TYPE_NVFP4`（原缺失）

验证结果：
- test-backend-ops：`./build-hip/bin/test-backend-ops test -b ROCm0 -p nvfp4` → 179/179 通过；`-p "mxfp4|rocmfp"` → 243/243 通过
- A/B 基准（Qwen3.5-0.8B-NVFP4, 1卡）：prefill pp512 4505→5175 t/s（+14.9%），decode 持平
- 注意：后端名是 `ROCm0`（不是 HIP），类型名小写 `nvfp4`

## 任务二：Laguna-S-2.1-NVFP4 跑 4 卡【进行中，卡在转换脚本】
### 已定位的根因
- 下载版 /opt/LLM/hf/jcbtc/Laguna-S-2.1-NVFP4.gguf（67GB）**格式不兼容**：
  - 该文件 scale 是 **e8m0 指数**（NVIDIA ModelOpt 打包，vLLM/TRT-LLM 专用），llama.cpp 规范的 NVFP4 是 **e4m3 scale**（上游 PR #19769 与本 fork 一致）
  - fork 按 e4m3 反量化 → 权重放大几千倍（median 60 vs 应 1.5e-5）→ logits 全 NaN → 采样退化恒定输出 token 14 `〈|` → 死循环、界面无输出
  - 模型卡原文："Llama.cpp (BF16 and Q4_K_M only)"
- 附加问题：原 GGUF 的 `laguna.attention.head_count` 只有单值 48，实际模型 **sliding 层是 72 heads**（q_proj [3072, 9216]），需要 per-layer 数组 [48,72,72,72,...]
- 验证工具：/tmp/tokgen2（打印 logits 前 5 + NaN 计数）、/tmp/tokgen（逐 token 打印）

### 原始权重
- /opt/LLM/hf/Laguna-S-2.1/：46 个 BF16 safetensors 分片（230GB）
- config：48 层（层0 dense + 47 MoE×256 experts/10 used）、head_dim 128、n_head 48/72（full/SWA，period 4）、gating per-head、sliding_window 512、rope full=yarn(θ500k, factor128, orig 8192)/swa=默认(θ10k, 128维)、norm_topk_prob、shared expert(1024)
- 张量命名：`model.layers.N.self_attn.{q,k,v,o}_proj/g_proj/q_norm/k_norm`、`mlp.{gate|gate_proj|up_proj|down_proj}`、`mlp.experts.{e}.{gate|up|down}_proj`、`mlp.shared_expert.{gate|up|down}_proj`、`mlp.experts.e_score_correction_bias`、无 input_scale

### 转换脚本（2026-08-01 已修复并验证，正在跑）
- 主脚本：`tools/convert-laguna-stream.py`（新写，流式：metadata 从原 GGUF 复制 → 逐张量 mmap 读 safetensors → 转置 → torch 量化 NVFP4 → 直接流式写 GGUF，无 BF16 中间文件）
- 量化器：`tools/nvfp4_quant.py`（torch 实现，**已验证与 ggml-quants.c 的 quantize_row_nvfp4_ref 逐字节一致**，验证脚本 /tmp/verify_quant.py）
- 已解决的坑：GGUFWriter 状态机顺序（先 add 全部 kv/tensor 再写文件）、张量数据顺序必须与 tensor info 顺序一致（pass2 按 order 逐个 mmap 读）、safetensors [out,in]→GGUF [in,out] 转置、expert 张量跨 shard（按 (层,kind) 收集 256 个再 stack）、1D 张量 shape 嵌套
- **上次卡点根因（已定位，不是 gguf-py 表的问题）**：`quant_shape_from_byte_shape` 报 "bytes per row (27) not a multiple of NVFP4 type size (36)" 的真凶是 **attn_gate（g_proj）**：safetensors 形状 [48|72, 3072] → 转置后 GGUF 行长为 48/72，不是 64 的倍数 → byte_shape[-1]=27/40.5。gguf-py 的 GGML_QUANT_SIZES[NVFP4]=(64,36) 是对的。这类张量**物理上不能**按 NVFP4 存储（ggml_row_size 有 `assert(ne % blck_size == 0)`），必须不量化
- **本次修复清单**（全部对照参考 GGUF /opt/LLM/hf/jcbtc/Laguna-S-2.1-NVFP4.gguf 的 814 个非 scale 张量逐一核对）：
  1. 不量化集合：1D 张量（norm/q_norm/k_norm/bias）→ F32；attn_gate（48/72 行）→ F32；ffn_gate_inp（router，参考文件就是 F32，保护 top-k 选择）→ F32。其余 2D/3D 全量 NVFP4
  2. **shared_expert 映射 bug**：原脚本把 `mlp.shared_expert.{gate|up|down}_proj` 映射成 `blk.N.ffn_{gate|up|down}.weight`（错误，MoE 层没有这些），实际应为 `blk.N.ffn_{gate|up|down}_shexp.weight`（图必需，缺失会加载失败）
  3. **exp_probs_b 命名 bug**：应为 `blk.N.exp_probs_b.bias`（无 ffn_ 前缀；图不引用它，但参考文件有此张量，保留对齐）
  4. **expert stack 维度**：`torch.stack(parts, dim=0)` → [256, out, in]（内存序 = GGUF 行主序 ne0=in 最快）；原 dim=-1 得到 expert 最快的错误字节序
  5. **合成 header 键**：GGUFReader 把 header 字段以 `GGUF.version/tensor_count/kv_count` 合成进 r.fields，复制 kv 时必须跳过（否则新文件 kv 区重复键，gguf-py 读取直接报错）
  6. head_count 覆盖用 `add_key_value(..., ARRAY, sub_type=INT32)`（原 add_array 无 sub_type 会炸）
  7. 数据段改用 writer 自带 `write_ti_data_to_file()` + 逐张量 `w.write_tensor_data()`（自动对齐、断言 nbytes、按序 pop），弃用手动 'ab' append
  8. 完整性校验：`assert len(order) == 814`（= 参考文件 1096 张量 - 282 个 e8m0 scale/input_scale）
- **早期验证（已通过）**：/tmp/verify_partial.py（手写 GGUF header/tensor-info 解析，GGUFReader 会急切读全部张量数据、部分文件会炸）：blk.0/1 的 F32 张量 vs 源 BF16 **逐位一致（max_rel=0）**；NVFP4 张量文件字节 vs 重新量化 **0/10616832 字节差异**；反量化误差 mean_rel~0.24 属 4-bit 量化正常水平（近零值拉高）
- 运行：后台任务，~1500% CPU + ~60GB RAM（大专家张量量化峰值）。141 个 453MB 专家张量，预计 8-12h
- 注意：safetensors `framework='np'` 不支持 bfloat16（TypeError），必须 framework='pt' + .float()
- 目标输出：/opt/LLM/Laguna-S-2.1-NVFP4-new.gguf（~67GB），跑通后 4 卡验证：
  `HIP_VISIBLE_DEVICES=0,1,2,3 LD_LIBRARY_PATH=build-hip/bin ./build-hip/bin/llama-bench -m /opt/LLM/Laguna-S-2.1-NVFP4-new.gguf -ngl 99`
- 参考：下载版已验证能加载（llama-bench 4 卡 pp64 62 t/s / tg16 31 t/s，只是权重反量化坏）

### 2026-08-01 晚：类型 id 发现（重要）+ 转换重启
- **发现 fork 的 gguf-py 与 C 枚举编号不一致**（文件内类型 id 以 C 为准，加载直接映射无转换表）：
  | 类型 | fork C (ggml.h) | gguf-py（已修） | 上游 |
  |---|---|---|---|
  | MXFP4 | 39 | ~~42~~→39 | 38 |
  | NVFP4 | 40 | ~~39~~→40 | 39 |
  | Q1_0 | 41 | ~~40~~→41 | 40 |
  | Q2_0 | 42 | ~~41~~→42 | 41 |
  - 已修 `gguf-py/gguf/constants.py`（3 行，含注释说明）。**若没修，gguf-py 写出的 NVFP4(id 39) 会被 C 读成 MXFP4，整个文件报废**
  - 推论：下载版官方文件的专家张量 id=40，fork C 一直按 NVFP4(e4m3) 解码 → e8m0 数据当 e4m3 → NaN。解码错配，非类型号冲突
  - 此问题的完整分析 + "原生支持官方 e8m0 格式"的工作计划见 **WORK_PLAN_OFFICIAL_NVFP4.md**
- 转换已因 id 修复**重启**（删掉旧文件），任务 ID bltunrh5o，日志 /tmp/laguna_convert.log
- 运行中 1500% CPU + ~60GB RAM；141 个 453MB 专家张量，预计 8-12h 完成（22:00 启动 → 明早 6-10 点）
- 验证脚本已同步更新为 id 40：/tmp/verify_partial.py、/tmp/verify_exps_chunked.py（都用 GGMLQuantizationType 枚举）

### 明日验证清单（按序）
1. `python /tmp/verify_partial.py` — 完整文件全部 814 张量：F32 逐位一致 + NVFP4 反量化误差（前几次跑过 blk.0/1，通过）
2. `python /tmp/verify_exps_chunked.py 1 down_proj`（可加参数验多层/多 kind）— 专家张量字节 vs 独立重量化（验证 stack dim=0 修复；分块防 OOM）
3. `python /tmp/inspect_official_nvfp4.py` — **官方格式解码方案确认**（KV 表 vs e2m1 表 × bias 127/128 + scale 粒度）→ 决定是否走 WORK_PLAN_OFFICIAL_NVFP4.md 的路线
4. 4 卡基准：`HIP_VISIBLE_DEVICES=0,1,2,3 LD_LIBRARY_PATH=build-hip/bin ./build-hip/bin/llama-bench -m /opt/LLM/Laguna-S-2.1-NVFP4-new.gguf -ngl 99`
   参考值：下载版 pp64 62 t/s / tg16 31 t/s
5. logits 健康检查：`/tmp/tokgen2 新模型 99`（无 NaN、前 5 logits 分布正常）——对比下载版的恒定输出 token 14
6. 磁盘：验证通过后删下载版原 GGUF（67GB）腾空间

## 验证方法备忘
- test-backend-ops 过滤器：`-b ROCm0 -p nvfp4`（类型名小写）
- 量化一致性：/tmp/verify_quant.py（对比 torch 量化 vs llama-quantize C 版输出，逐字节）
- token 诊断：/tmp/tokgen2 模型 99（GPU logits 检查）、`/tmp/tokgen 模型 0`（CPU 采样）
- 磁盘：/opt/LLM 当前 ~608G 可用；转换完建议删下载版原 GGUF（67GB）腾空间



候选方向：decode 路径（mmvq vec_dot_nvfp4_q8_1 的 VDR 微调受 qi/vdr 约束不可直接提，但可以试 half2 累加）

## 任务三：官方 ModelOpt NVFP4 GGUF 原生支持【已完成并验证，2026-08-01 深夜】

### 格式破译结论（inspect 脚本 + 与 BF16 源逐值验证）
官方 /opt/LLM/hf/jcbtc/Laguna-S-2.1-NVFP4.gguf（67GB）专家张量（type 40）与 fork NVFP4 **同构**：
- 64 值块 = [4B scale][32B data]，data 每 16 值 8B 交错 nibble（与 block_nvfp4 完全相同）
- **唯一差异 = scale 编码**：官方 = 8 位定点对数 `scale = 2^((b-169)/8)`（KV 表语义，等价 e2m1 原值 ×2^((b-161)/8)），每 16 值 1 字节，b ∈ 1..126
- 值表 = e2m1 {0,0.5,1,1.5,2,3,4,6}（= fork KV 表 ×2）；nibble bit3=符号、bits2-0=表索引
- **排除的假设**：不是每 64 值 2B scale（实际 36B/64 与 fork 同）、无转置写（GGUF 数据本就 out 主序 = 标准）、e4m3 转换不可行（动态范围 [2^-10,2^8] 装不下官方 [2^-21,2^-5.4]，主流段 b 97-113 全落次正规区误差 50-80%）
- 全层验证：27 个 (层,kind) 解码 corr 0.99+，mean_rel 0.2-0.9（正常 4-bit 水平）

### 实现（20 个文件，全部验证通过）
1. 新类型 `GGML_TYPE_NVFP4_E8M0 = 43`（块结构复用 block_nvfp4）
2. CPU：dequantize_row_nvfp4_e8m0（scale = exp2f((b-169)*0.125f)）+ quantize ref + type_traits + vec_dot
3. CUDA/HIP：common.cuh helper + type_traits 特化、vecdotq/mmvq/mmq/load-tiles/convert 的 scale 解码替换 + 全部分支、gfx906 MMQ config、GET_ROWS 表
4. 加载重映射（llama-model-loader.cpp）：架构 laguna + 存在 `ffn_*_exps.scale` 记账张量 → type-40 专家张量重映射为 NVFP4_E8M0（数据零修改，加载零额外开销，nbytes 相同）
5. gguf-py：NVFP4_E8M0 = 43 + QUANT_SIZES (64,36)；test-backend-ops 类型列表

### 验证结果
- test-backend-ops：CPU `-p nvfp4` 444/444 ✓；ROCm0 328/328 ✓（含 MUL_MAT type=nvfp4_e8m0）
- 4 卡基准：`HIP_VISIBLE_DEVICES=0,1,2,3 llama-bench -m 官方文件 -ngl 99 -r 1` → **pp512 308.2 t/s / tg128 32.5 t/s**（117.56B 正常识别，之前误解码 pp64 仅 62 t/s）
- logits 健康：/tmp/tokgen2 → top0="ĊĊ"(19.0) top1="ĠWhy" top2="ĠThe"，**NaN 0 / Inf 0**（对比之前全 NaN 恒定 token 14）
- 附注：转换路线（convert-laguna-stream）已不需要；/opt/LLM/Laguna-S-2.1-NVFP4-new.gguf（129MB 中断残留）可删

## 任务四：官方文件推理乱码排查【进行中，2026-08-02 暂停存档】

### 现象
- 官方文件 + server/llama-cli 带模板 prompt：**输出乱码**（无 NaN，但 token 无意义，之前还出现过"Hello 回声循环"）
- **对照**：unsloth 版 `/opt/LLM/hf/unsloth/Laguna-S-2.1-MXFP4_MOE-00001-of-00003.gguf`（MXFP4，3 分片）**同一 server 配置推理正常**（用户确认）

### 已排除（全部验证过）
1. 旧二进制（build/bin 17:09 无 NVFP4_E8M0）→ 已重建（build/ + build-hip 均 23:18+，strings 确认 21 处类型引用）
2. `-sm tensor`（AllReduce init failed → butterfly fallback → GGML_ASSERT 崩）→ 改 `-sm layer`
3. `-fa 1`（tokgen3 测过 fa 健康）、`-c` 8192/262144、`-b/-ub` 512/1024、采样参数（对齐 tokgen3 后仍乱）
4. `--kv-unified`、`--cache-reuse`/`--context-shift`（日志显示已被自动禁用）、`LLAMA_ENABLE_MTP_OPT`（源码无此变量）
5. chat template：v8（GGUF 内置）与 v24（chat_template_laguna.jinja）都乱；**关键发现：任何 `<system>` 块（含空块）→ 乱码**（P1 无 system 健康 / P1c/P1d 带 system 乱）→ 已把 v24 模板默认 system_message 改为空（README 证实"模型训练时无 system message"）——**但 v24 空 system 仍乱**
6. tokenizer/vocab：两文件特殊 token id 相同（〈|EOS|〉=2、<assistant>=23、<think>=18）
7. hparams：head_count 72/8、sliding 512、rope 500000/64+10000/128、expert 配置（256/10/2.5/norm=True/dense_lead=1）**两文件完全相同**；yarn_attn_factor 不同（jcbtc 1.3466 vs unsloth 1.0 vs HF 1.4852）但 **llama.cpp 不读该 kv**（grep 无引用）
8. 非专家张量：**431 个 BF16/F32 张量 vs safetensors 全部 ratio=1.0 逐位一致**（attn/router/shexp/输出层/dense 层）
9. exp_probs_b：全 0（无影响）；input_scale：per-expert 激活 scale（W4A4 专用，BF16 激活不需要，vLLM Marlin 也删除）；GPU vs CPU 推理结果相同（排除内核）

### 关键发现（专家权重语义）
- jcbtc 专家权重 vs safetensors：**差 scale[e] × K**——`scale[e]` = 每 expert 权重 scale（`blk.*.ffn_*_exps.scale` [256] F32，llama.cpp 已加载但 **laguna 的 build_moe_ffn 未使用**）、`K ≈ 18479`（全局常数，最小二乘拟合 spread <1%）
- **已修**：`src/models/laguna.cpp` 的 build_moe_ffn 调用补传 `ffn_up_exps_s/gate_exps_s/down_exps_s`（图乘 scale[e]，cohere2moe 同模式）→ logits 变化（'ĠWhy' → 'Hello 回声'）但**仍乱**
- K 尝试：moe_out × 18479（被 pre-norm 吸收，logits 逐值不变=无效）；scale 字节 +113 偏移融 K（也乱）；scale 张量 ×K（也乱）——**K 被层 norm 吸收，确认与乱码无关**
- unsloth 专家（MXFP4）vs safetensors：**ratio ≈ 0.97**（无 K 归一化，直接可推理）——与 jcbtc（1/K）形成对照
- **未解决矛盾**：jcbtc 权重"修复到差全局 K"（理论可推理）但仍乱；unsloth 正常

### 未提交改动（工作树）
- 任务三的 NVFP4_E8M0 支持（20 文件）+ 任务四的 laguna.cpp 修复（scale 张量传参 + moe_out × K）
- `tools/fix-official-nvfp4-scale.py`（当前版本：scale 张量 × K；历史版本：scale 字节偏移）
- `/opt/LLM/Laguna-S-2.1-NVFP4-fixed.gguf`（scale 张量 ×K 版修复文件）
- 注意：`moe_out × 18479`（laguna.cpp）疑似无效改动，可撤销

### 下一步候选（未做）
1. **重量化替换测试**（决定性）：用 safetensors 标准重量化 1 个专家张量替换进 jcbtc 文件 → logits 若变且正常 = 权重字节问题；不变 = 图/配置问题
2. 逐层输出对比 jcbtc vs unsloth（找第一个分歧层）
3. unsloth MXFP4 块布局是 0.6875B/值（非 17B/32），GGUF 间专家直接对比未做成
4. 兜底：若 jcbtc 修不好，unsloth 文件可直接用（推理正常）

### 排查工具（/tmp）
tokgen3（多步生成监控，支持 special/n_ctx/prompt/kv-override 参数）、tokgen5（特殊 token id）、tokgen6（minja 模板渲染）、tokgen7（GGUF 张量对比，API 不存在未编译）、verify_full_tensor.py / fit_K_exact.py / check_per_expert.py / scan_all_bf16.py / check_inputscale_ratio.py 等


## 任务四：DeepSeek-V4-Flash-0731 ROCMFPx (type 107 = Q2_0_ROCMFPX affine) 移植【已完成并验证 2026-08-03】

### 背景
- 模型 /opt/LLM/hf/deepseek/DeepSeek-V4-Flash-0731-Abliterated-ROCMFPx-Strix-Lean-2.58bpw.gguf（91GB，HF 下载）报 "tensor 'blk.0.ffn_down_exps.weight' has invalid ggml type 107"
- 129 个 MoE 专家张量用 type 107（2.50 bpw，10B/32 权重），其余 574×101 + 43×100 + F32 等 fork 已支持
- **格式真相（作者 glovepost 在 HF 讨论区披露）**：不是 ROCmFPX 仓库的 S40 码本格式，而是 **affine fp2**（作者引擎 ember 的 "Q2_0_ROCMFP2"）：
  - 块布局与 block_rocmfp2 字节级相同（qs[8] + e[2]，位打包顺序一致）
  - e[0] = scale（全部 32 权重）、e[1] = offset（全部 32 权重），均 UE4M3 编码
  - code 是字面值 c ∈ {0,1,2,3}，value = c*scale - offset
  - 曾误按 S40 MORD/MSM 码本解码（输出乱码），affine 后正常

### 移植内容（21 文件，全部在 ggml/rocmfpx/ + ggml/src/ + llama 侧）
- ggml.h: GGML_TYPE_Q2_0_ROCMFPX=107, COUNT=108, GGML_FTYPE_MOSTLY_Q2_0_ROCMFPX=112
- rocmfpx.c/.h: block_rocmfp2 + affine 量化/反量化/row_size（quantize 用 min/max → scale=(max-min)/3, offset=-min）
- ggml.c: type_info + ftype 映射 + quantize_chunk；ggml-quants.c: 校验 case + wrapper
- ggml-cpu: quants.c vec_dot（sumc = Σc·q8, sumq = Σq8, result = scale·sumc − offset·sumq）+ type_traits + ops.cpp
- ggml-cuda: common.cuh traits、dequantize.cuh、vecdotq.cuh（v_perm_b32 表 0x03020100 字面码 + MMVQ/MMQ dp4a，Σq8 用 dp4a(0x01010101,u) 技巧）、mmq-load-tiles/mmq-vec-dot/mmq.cuh/mmq.cu/mmvq.cu dispatch、gfx906 config、mmq-instance-q2_0_rocmfpx.cu
- getrows.cu: **新增 ROCmFP 全家 GET_ROWS**（DeepSeek token_embd 是 101 非 KAT 的 Q6_K，原 fork 缺口；get_rows_cuda_q 用 float2 风格 dequantize）
- llama: llama.h/llama-quant.cpp/llama-model-loader.cpp ftype 映射、quantize 工具表、gguf-py 常量
- test-backend-ops: q2_0_rocmfpx 加入 MUL_MAT 用例（22 个形状）

### 验证
- test-backend-ops: ROCm0 110/110 rocmfp（含 22 个 q2）+ CPU 22/22 通过
- 4 卡推理正常：Prompt 31.3 t/s / Generation 24.6 t/s（与作者公布 ~23-24 tok/s AR 一致）
- 输出连贯（思考模式正常）

### 环境坑（重要）
- ~/.bashrc:175 `export HIP_VISIBLE_DEVICES=0,1` 限制只可见 2 卡（HIP 0=物理4, HIP 1=物理3）；**HIP 与 rocm-smi 编号顺序相反**
- KAT server（PID 7484, kat.sh HIP_VISIBLE_DEVICES=1）常驻物理 3（31.8GB）
- 跑 DeepSeek 用 HIP_VISIBLE_DEVICES=0,2,3,4（物理 4,2,1,0 全空闲）；-sm layer 4 卡
- llama-cli 输出巨大，验证时重定向文件或用 tail

### 后续优化候选（作者讨论要点）
- 融合 MMVQ/MMQ kernel 已就位（affine 内层）；作者还提 LUCE_MMVQ_MAX_NCOLS=4 调优 knob
- DSpark 投机解码（Lucebox/DeepSeek-V4-Flash-DSpark-Drafter-GGUF，Q4RMFP4 denseF16，sha256 48883d35...）可上 ~32 tok/s；draft 校准 4 routed experts（模型默认 6，改 4 可 +36% decode）
- 作者公开数字 248-253 tok/s prefill 是 gfx1151 专用融合 kernel 测的，MI50 无参考
