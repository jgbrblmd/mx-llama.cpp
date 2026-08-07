# 工作计划：原生支持官方 ModelOpt NVFP4 格式（e8m0）

目标：让 fork 直接加载官方文件 `/opt/LLM/hf/jcbtc/Laguna-S-2.1-NVFP4.gguf`（67GB），
免去 8-12h 的全量重新量化。官方模型卡明示仅支持 vLLM/TRT-LLM/Transformers（NVIDIA 栈），
llama.cpp 侧只承诺 BF16/Q4_K_M——因为官方打包格式与本 fork 的 NVFP4 规范不同。

## 一、格式事实（2026-08-01 已勘察）

### 官方（ModelOpt）专家张量布局
- 每 64 值 = **16B fp4 数据 + 2B e8m0 scale**（18B/64 = 2.25 bit/值）
- fp4 数据 = 4-bit e2m1（值表待 inspect 脚本确认：llama KV 表 {0,1,2,3,4,6,8,12} vs NVIDIA e2m1 {0,0.5,1,1.5,2,3,4,6}）
- e8m0 scale = 纯指数无尾数，`scale = 2^(e - bias)`，bias 127 或 128（待确认）
- scale 粒度：每 64 值 1 个，还是每 256 值 1 个（4 块共享，待确认）
- 非专家张量：attn/mlp/embedding 为 **BF16**（fork 已能加载），norm/gate/router 为 F32
- 附加 282 个 `ffn_*_exps.scale` / `.input_scale` 张量（ModelOpt 记账，图全部 TENSOR_NOT_REQUIRED）

### 本 fork 的类型编号（关键发现，已修正 gguf-py）
| 类型 | fork C (ggml.h) | 原 gguf-py（已修） | 上游 llama.cpp |
|---|---|---|---|
| MXFP4 | 39 | ~~42~~ → 39 | 38 |
| NVFP4 | 40 | ~~39~~ → 40 | 39 |
| Q1_0 | 41 | ~~40~~ → 41 | 40 |
| Q2_0 | 42 | ~~41~~ → 42 | 41 |

- fork 的 C 枚举因自带 MXFP4=39 而整体后移 1，**文件内的类型 id 以 C 枚举为准**（加载是直接映射，无转换表）
- 已修：`gguf-py/gguf/constants.py` 对齐 C（否则 gguf-py 写出的文件 C 侧读成 MXFP4）
- 推论：官方文件的专家张量 id=40，fork C 目前按 **NVFP4(e4m3, 36B/64)** 解码 → e8m0 数据被当 e4m3 → 权重放大几千倍 → NaN。**这是解码语义错配，不是类型号冲突**
- 因此原生支持 = 新增一个 e8m0 变体类型 + 加载时按启发式重映射，不需要动类型 id

## 二、方案选择

### 方案 A：运行时 e8m0 类型（本计划主线）
新 ggml 类型 + 内核 + 加载重映射，官方文件原样加载。

### 方案 B：格式迁移工具（备选，工作量小得多）
写一个流式工具：读官方 GGUF → 只解码 141 个专家张量（32GB）→ 用现有 nvfp4_quant.py
重新量化为 e4m3 NVFP4 → 重写文件。attn 等 BF16 保持原样。产出 ~67GB 标准文件。
- 优点：零内核改动、零检测逻辑，复用已验证的量化器；1-2h 计算 + ~半天工具
- 缺点：仍是"转换"路线（虽然比全量重量化快一个量级），产出的专家是 4.5-bit（精度比官方 2.25-bit 好）

### 建议
先跑 inspect 脚本确认官方格式细节；若确认干净（解码后与 BF16 源吻合），
**方案 A 与方案 B 并行可行性评估，推荐 B 先落地**（官方专家 2.25-bit 精度损失明显，
B 产出 4.5-bit 专家 + BF16 attn 的组合在精度上全面优于官方原文件，且不需要动推理栈）。
方案 A 保留为长期目标（若后续 ModelOpt 文件成为常态格式）。

## 三、方案 A 实施步骤

### 1. 前置确认（inspect 脚本，秒级）
`/opt/venv/common/bin/python /tmp/inspect_official_nvfp4.py`
- nibble 直方图落在 {0,1,2,3,4,6,8,12} → fp4 确认
- 4 种解码组合（KV 表 vs e2m1 表 × bias 127/128）与 BF16 源对比 → 选出正确的表与偏置
- scale 重复周期 → 每 64 还是每 256 值一个 scale

### 2. 新类型
- `ggml/include/ggml.h`：`GGML_TYPE_NVFP4_E8M0 = 43`（43-99 空闲）
- `ggml/src/ggml.c`：type_traits 表（blck_size=64, type_size=18, dequant 函数指针）
- 注意 `ggml_row_size` 的 `ne % 64 == 0` 断言：专家张量行长为 1024/3072 ✓ 无影响

### 3. 反量化/量化（ggml-quants.c）
- `dequantize_row_nvfp4_e8m0`：块 = 2B e8m0 → scale = 2^(e-bias)；16B fp4 → nibble 高低 → 值表 ±
- `quantize_row_nvfp4_e8m0_ref`：per-64 块 amax → e8m0（2^(ceil(log2(amax))-bias) 取整）+ 最近值编码（供测试/工具用，推理不需要）
- 若 scale 为每 256 值共享：块定义改 (256, 72)，数据按 4×64 连续

### 4. ROCm/CUDA 内核（比现有 NVFP4 更简单：每 64 值 1 个 scale）
参考文件：`ggml/src/ggml-cuda/{vecdotq.cuh, mmq-load-tiles.cuh, mmq.cuh, mmvq.cu, convert.cu}`
- mmvq：scale 加载从 4×(e4m3) 改为 1×e8m0 扩展；fp4 数据解码逻辑不变（同 KV 表路径）
- mmq/vecdotq：同步简化
- `ggml-cuda.cu` supports_op/type 支持表补新类型

### 5. 加载重映射（检测）
位置：`src/llama-model-loader.cpp`（tensor info 解析后、create_tensor 前）
启发式（对目标文件可靠）：
- 架构 == laguna 且文件含 `ffn_*_exps.scale` 张量（ModelOpt 记账特征）→ 该文件的 type-40 专家张量重映射为 NVFP4_E8M0
- 备选：启动参数 `--model-format nvfp4-e8m0`（通用兜底）
注意：fork 中 id=41 才是真 Q1_0，type-40 在 fork 语义里就是 NVFP4——重映射不与其他格式冲突

### 6. gguf-py
- `gguf-py/gguf/constants.py`：`NVFP4_E8M0 = 43` + GGML_QUANT_SIZES 条目 (64, 18)（工具链完整性）

### 7. 验证
- `test-backend-ops test -b ROCm0 -p nvfp4_e8m0`（类型名小写规则）
- 加载：`llama-bench -m /opt/LLM/hf/jcbtc/Laguna-S-2.1-NVFP4.gguf -ngl 99`（4 卡）
- logits 健康检查：`/tmp/tokgen2`（无 NaN、分布正常）——对比 BF16 源模型 logits 相关性
- 质量对比：解码后专家权重 vs BF16 源（全张量 mean_rel）——预期 ~0.2-0.3（2.25-bit 固有损失，比 4.5-bit 的 ~0.24 差多少，量化评估）

## 四、风险与待确认项
- [ ] fp4 值表约定（KV vs e2m1）——inspect 脚本
- [ ] e8m0 偏置（127 vs 128）——inspect 脚本
- [ ] scale 粒度（64 vs 256）——inspect 脚本
- [ ] 官方专家 2.25-bit 的精度损失是否可接受（与重新量化的 4.5-bit 对比）
- [ ] 检测启发式的误伤面（非 laguna 架构的 type-40 文件不受影响 ✓）

## 五、工作量估计
- 方案 A：1-2 天（类型 + CPU 编解码 ~0.5 天，ROCm 内核 ~0.5-1 天，重映射+验证 ~0.5 天）
- 方案 B：~1 天（工具 ~0.5 天，计算 1-2h）
