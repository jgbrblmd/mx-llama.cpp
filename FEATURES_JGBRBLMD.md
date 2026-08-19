# jgbrblmd/mx-llama.cpp Fork Features

This fork (based on jgbrblmd/mx-llama.cpp) extends mx-llama.cpp with additional features and optimizations.

## Key Differences from mx-llama.cpp

### Quantization Enhancements
- **ROCmFP4 STRIX Support**: Native ROCmFP4 quantization for AMD GPUs, enables efficient FP4 inference
- **Qwen3Next MLX Handling**: Improved support for Qwen3Next models from MLX format
- **Qtype 105/106 Per-Expert Mix**: Support for Lucebox DSV4-Flash MIX-STRIX mixed quantization models

### CUDA Optimizations
- **q8_repack System**: Complete CUDA q8 repack infrastructure with:
  - Async weight uploads
  - Device-side repacking for tensor-parallel models
  - Pipeline repacked tensor-parallel loads
  - Extra buffer type management

### Architecture Support
- **Tensor Parallelism Improvements**: Split attention output weight at GQA-group granularity
- **Draft Model Loading**: Load draft models without extra buffer types
- **Graph Placement**: Write-only graphs placed on stage owning the buffer

### Stability Fixes
- Null vbuffer segfault fix after failed reserve
- Null-task assert fixes in server
- ne[0] % blck logic check fix
- Enhanced CGN card compatibility guards

## Build & Usage

Same build process as mx-llama.cpp. All features enabled by default.

For mixed quantization models, use appropriate converter tools to prepare model files.

## Relationship to Upstream

- Based on mx-llama.cpp (mxxm-t/mx-llama.cpp)
- Periodically syncs with upstream ggml-org/llama.cpp
- Adds features not present in either upstream or mx-llama.cpp
