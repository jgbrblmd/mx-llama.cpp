# jgbrblmd-sync 移植进度

**日期：** 2026-08-24  
**当前分支：** jgbrblmd-sync（基于 mxxm-t/master 的干净分支）  
**远程 fork：** jgbrblmd/mx-llama.cpp（fork/master）

## 分支关系

```
mxxm-t/master (30d703351)  ←─ jgbrblmd-sync (ef03a6173) + 44 commits
fork/master (330ee780b)  — 独立分支，+ 47 commits（44 重叠 + 3 上游 PR）
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

## 尚未移植的 3 个上游 PR（仅在 fork/master 上）

1. **#26389** — 投机解码计数器到 `/metrics` 端点
2. **#26510** — 投机解码配置重构（common_speculative_init）
3. **#26814** — 从 draft GGUF metadata 自动检测 spec type

## 编译状态

- 所有冲突标记已解决
- 编译测试：cmake --build . --parallel 32（进行中）
- 预计编译时间：~5 分钟

## 下一步

1. 完成编译测试，验证无错误
2. 考虑移植 3 个上游 PR（投机解码改进）
3. 功能验证测试
