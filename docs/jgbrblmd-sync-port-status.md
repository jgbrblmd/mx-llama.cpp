# jgbrblmd-sync 移植进度

**日期：** 2026-08-24  
**当前分支：** jgbrblmd-sync（基于 mxxm-t/master 的干净分支）  
**远程 fork：** jgbrblmd/mx-llama.cpp（fork/master）

## 分支关系

```
mxxm-t/master (def64b335)  ←─ jgbrblmd-sync (a76d48d01) + 44 commits
fork/master (330ee780b)    — 独立分支，+ 47 commits（44 重叠 + 3 上游 PR）
```

## 已移植功能（44/47 commits）

jgbrblmd-sync 包含以下完整功能：
- ROCmFP4/FP6/FP8 量化类型支持（Q2_0_ROCMFPX, Q3_0_ROCMFPX）
- gfx906 GPU 内核优化（NVFP4 unroll, MMQ 8-warp tile）
- DFlash 2 投机解码支持
- qtype 105/106 per-expert mix 支持（Lucebox DSV4-Flash MIX-STRIX）
- Muse-glimmer 模型支持
- Bailingmoe3 支持
- 各种 bug 修复和稳定性改进

## 与 mxxm-t/master 的同步状态

已将 mxxm-t/master 的 22 个 commits（def64b335）合并到 jgbrblmd-sync，包括：
- 投机解码 API 变更（llama_sampler_i 新增 backend_reset/copy_state）
- llama_load_mode 新增 AUTO 选项
- llama_context_params 格式化调整
- ggml-backend 元数据改进
- CUDA/HIP 内核优化（workspace splitting, BLAS recovery 等）
- DSpark 多序列支持（替代原 DFlash2 单序列实现）
- Kimi-K3 / dots3-note / Ling-3.0-flash 模型类型

## 编译状态

### 核心库（llama）
- ✅ 编译成功，66 个 target 构建完成
- llama-simple, llama-batched, mtmd 等均正常

### llama-server
- ✅ 编译成功，所有 63 个错误已修复
- 修复内容：
  - 移除 `using json = nlohmann::ordered_json`（改用 common_json）
  - 删除重复的 `server_metrics` 定义（已在 server-common.h 中）
  - 替换 `result_timings get_timings()` → 使用 `slot.stats` 构造
  - 更新 metrics 方法调用：`on_prompt_eval/on_prediction/on_decoded` → `add_prompt/predict.add`
  - 修复 metrics 字段命名：新 API 使用 bucket 结构（prompt.count, predict.time 等）
  - 修复 `has_media()` → `has_mtmd`
  - 修复 `eval_llama_cmpl_schema` 调用参数顺序
  - 修复 JSON brace 初始化兼容性问题
  - 修复 `/slots` 端点使用错误的结果类型（改为 server_task_result_slots）
  - 添加缺失的 `common_get_env` 声明到 common.h

### 未移植的 3 个上游 PR（仅在 fork/master 上）
1. **#26389** — 投机解码计数器到 `/metrics` 端点
2. **#26510** — 投机解码配置重构（common_speculative_init）
3. **#26814** — 从 draft GGUF metadata 自动检测 spec type

## 功能文档

详细功能分类见 [jgbrblmd-sync-features.md](./jgbrblmd-sync-features.md)，按以下模块组织：
ROCmFPX/ROCmFP4 量化类型、Lucebox per-expert mix、gfx906 内核优化、模型支持（Muse/Bailing/DFlash2）、投机解码、GDN 修复、转换器工具、服务端改进。

## 下一步

1. ✅ 修复 server-context.cpp 编译错误（已全部完成，见 commit 6d5c12ac0）
2. 移植 3 个上游 PR（投机解码改进：#26389 / #26510 / #26814）
3. 功能验证测试

## 关键提交

- `a76d48d01` merge mxxm-t/master: sync upstream API changes and bug fixes
- `ef03a6173` gfx906 NVFP4: unroll k-loop for better prefill throughput
- `5a0ded4a4` dflash2: complete DFlash 2 support from PR #27342
