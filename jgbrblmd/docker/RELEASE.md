# jgbrblmd123/mx-llama.cpp:gfx906-rocm716

面向 AMD **gfx906**（MI50/MI60，32GB HBM）的 llama.cpp 高性能推理 Docker 镜像，
内嵌 **ROCm 7.16** 运行时（来自 `zenth815/rocm10-gfx906:sdk`，SDK 目录名 `/opt/rocm-10.1.0`）。

| 项目 | 值 |
|---|---|
| 镜像 | `jgbrblmd123/mx-llama.cpp:gfx906-rocm716`（~820MB，runtime-only） |
| 基座 | ubuntu:24.04 + 最小运行依赖（libgomp1/libdrm/numa 等） |
| 代码 | mx-llama.cpp `master` @ `37206c4e6`（含 upstream b10760 合并） |
| 构建脚本 | `../build-rocm716.sh`（`GGML_HIP=ON GGML_HIP_GRAPHS=ON GGML_HIP_RCCL=ON`，`-mllvm -amdgpu-sched-strategy=max-ilp`，gfx906） |
| ROCm 运行时 | 7.16，按二进制 ldd 传递闭包裁剪（26 个 so + rocBLAS gfx906 预编译 kernel），放在镜像内 `/opt/rocm-10.1.0`（与二进制 RUNPATH 一致，`/opt/rocm` 为软链） |

> 参考镜像 `mxxm/mx-llama.cpp:gfx906`（1.4GB，ROCm 7.2 运行时）结构相同；本镜像区别是 ROCm **7.16** 运行时 + 本分支的量化/性能特性。

## 分支功能特色

### KV cache 量化

- **TurboQuant 2/3/4-bit KV cache**（`turbo2`/`turbo3`/`turbo4`，ggml 类型 46/44/45，自 Wizard815/mx-llama.cpp-Rocm10 合入）
  - 3-bit = 2-bit PolarQuant + 1-bit QJL；4-bit = 3-bit + QJL；2-bit = 纯 2-bit PolarQuant
  - 基于 O(d log d) Walsh-Hadamard 旋转（`ggml_turbo_wht`）
  - 使用：`-ctk turbo4 -ctv turbo4`
  - 约束：**所有层必须在 GPU 上**（`-ngl 99`），且必须开启 **Flash Attention**（数据存储在 FWHT 旋转空间）
  - 可选 `TURBO_LAYER_ADAPTIVE=1` 按层自适应
- 常规低比特 KV：`-ctk q8_0 -ctv q8_0` 等（FP8 KV scale 张量转换时已剥离）

### 权重量化

- **CT_INT4**（compressed-tensors INT4，group_size=128 对称量化，ggml 类型 109 / ftype 113），带 CUDA/HIP 后端
- **ROCmFPX FP8**：`Q4_0_ROCMFP4` / `Q4_0_ROCMFP4_FAST` / `Q8_0_ROCMFPX`（含 get_rows 修复与转换脚本，用法见下文「5. ROCmFPX 量化脚本」）

### gfx906 专项优化

- K-quant（IQ4_NL / Q4_K / Q5_K / Q6_K / Q5_1）q8_repack 路径，全部 split 模式，**prefill 提速**
- K-quant 与 IQ4_NL 的 MoE up/gate mat-vec 融合（fused epilogue），**decode +6~8%**
- fattn vec kernel gfx906 修复（V_DOT2 + Q_q8_1）
- Q4_K/Q5_K MoE GEMM 寄存器溢出（spill）削减
- 2-GPU 张量并行（`-sm tensor`）

### 投机解码 / MTP

- MTP prefill graph shape 稳定化（qwen4exp，**prefill 2.5x**）
- MTP carrier 序列起点清零、recurrent snapshot ring 支持投机回滚、PLE 表预 fault（`-ff` 预加载 embedding）
- 实测（Tiel-Coder 35B-A3B Q4_K，2-GPU TP）：28.5 → **37.13 t/s**（基线 30.18）；MTP 头 + output.weight Q8_0 后 tg 31.2→33.1 / MTP 39.5→46.2 t/s
- `LLAMA_ENABLE_MTP_OPT=1`（镜像内已默认设置）

### 其他

- qwen4exp（NextN eh_proj 融合、`-sm tensor` 支持、compress_ratios 全层）
- MiMo2 滑动窗口 pattern 全层读取
- MoE 融合/确定性 top-k、fused top-k router 求和、meta graph 优化（zero-token graph 依赖、未分配 meta buffer 的节点跳过）
- server 端投机验证 token 概率返回

## 使用方法

### 1. 单机单卡

ROCm 容器需要 `kfd` + `dri` 设备（本机无 `render` 组，只加 `video` 即可）：

```bash
docker run -d --name llama \
  --device /dev/kfd --device /dev/dri --group-add video \
  -p 8080:8080 \
  -v /path/to/model.gguf:/model.gguf \
  jgbrblmd123/mx-llama.cpp:gfx906-rocm716 \
  llama-server -m /model.gguf --port 8080 -c 32768
```

多卡张量并行：

```bash
docker run -d --name llama --device /dev/kfd --device /dev/dri --group-add video \
  -p 8080:8080 -v /path/to/model.gguf:/model.gguf \
  -e HIP_VISIBLE_DEVICES=0,1 \
  jgbrblmd123/mx-llama.cpp:gfx906-rocm716 \
  llama-server -m /model.gguf --port 8080 -sm tensor
```

> 本机 GPU 编号对应（rocm-smi Device ↔ HIP_VISIBLE_DEVICES，**反向**）：
> Device 0↔3、1↔2、2↔1、3↔0。选卡以 `rocm-smi` 输出为准。

### 2. 常用特性开关

```bash
# TurboQuant 4-bit KV（省 ~6x KV 显存，需 -ngl 99 + fa）
llama-server -m /model.gguf -ngl 99 -fa -ctk turbo4 -ctv turbo4

# 8-bit KV
llama-server -m /model.gguf -ngl 99 -fa -ctk q8_0 -ctv q8_0
```

### 3. 镜像内置环境变量

| 变量 | 值 | 说明 |
|---|---|---|
| `LD_LIBRARY_PATH` | `/opt/rocm-10.1.0/lib:/usr/local/lib` | ROCm 7.16 + llama so |
| `ROCM_PATH` | `/opt/rocm` | |
| `HSA_OVERRIDE_GFX_VERSION` | `9.0.6` | |
| `HSA_FORCE_FINE_GRAIN_PCIE` | `1` | |
| `GPU_MAX_HW_QUEUES` | `8` | |
| `GGML_ENABLE_CUSTOM_AR` | `1` | |
| `LLAMA_ENABLE_MTP_OPT` | `1` | MTP 优化 |

### 4. 验证

```bash
docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
  -e HIP_VISIBLE_DEVICES=2 -v /path/to/small.gguf:/model.gguf \
  jgbrblmd123/mx-llama.cpp:gfx906-rocm716 llama-bench -m /model.gguf -n 32 -r 32
```

实测基线（OvisOCR2 0.8B Q8_0，单卡，无 TP）：`pp512 6877 t/s / tg32 198.5 t/s`，
设备识别 `gfx906:sramecc-:xnack- (0x906), VRAM 32752 MiB`。

### 5. 无损转换 HF CT_INT4 检查点（compressed-tensors INT4 → GGUF）

vLLM 用的 compressed-tensors INT4 检查点（`pack-quantized`、`num_bits=4`、`group_size=128`、对称、无 zero-point）
可以**无损**转成 GGUF：转换器直接读取 `*.weight_packed` + `*.weight_scale`，重打包为本分支的 `CT_INT4` 布局
（ggml 类型 109，68 字节 / 128 权重组），**不做任何重新量化**，数值与 vLLM 加载的 int4 码字完全一致。

示例（RedHatAI/Qwen3.8-27B-INT4，Qwen3.5 VLM 文本塔 + MTP 头，group_size=128 INT4）：

```bash
cd /opt/LLM/working/jgbrblmd/mx-llama.cpp
/opt/venv/common/bin/python convert_hf_to_gguf.py /opt/LLM/hf/RedHatAI/Qwen3.8-27B-INT4 \
  --outtype auto \
  --outfile /models/Qwen3.8-27B-CT_INT4.gguf
```

- 输出默认落在模型目录下（`<模型目录名>-<ftype>.gguf`），用 `--outfile` 指定更干净
- 自动识别 `model_mtp.safetensors` 中的 `mtp.*` 权重并映射到 nextn 层（MTP 投机解码开箱即用）
- vision 塔被量化 recipe 排除（保持 bf16），但主 GGUF 只含文本模型；需要图像输入时另跑 `--mmproj` 导出
- 未量化的 `output.weight`（lm_head）和 MTP 头每步 decode 全量读取，可用 `--ct-int4-head-type Q8_0`
  （默认，省 ~1.1 GiB 流量、max logit 误差 ~5e-4）或 `--ct-int4-mtp-type` 控制；`keep` 保持检查点原精度

用本分支镜像推理（27B int4 约 15 GiB，单张 32GB MI50 放得下）：

```bash
docker run -d --name llama --device /dev/kfd --device /dev/dri --group-add video \
  -p 8080:8080 -v /models/Qwen3.8-27B-CT_INT4.gguf:/model.gguf \
  jgbrblmd123/mx-llama.cpp:gfx906-rocm716 \
  llama-server -m /model.gguf --port 8080 -ngl 99 -c 32768
```

> 无损前提：检查点必须严格满足 `pack-quantized` + INT4 + group 128 + symmetric 检测条件
> （`conversion/base.py` 中 `quant_method == "compressed-tensors"` 判定）；不满足时回退为
> dequant → 按 `--outtype` 重新量化（非无损）。

### 6. ROCmFPX 量化脚本（bf16 GGUF → ROCmFP4 / Q8_0_ROCMFPX）

`tools/convert-qwen35moe-rocmfp4.py` 把 Qwen3.5 MoE/dense 的 **bf16 GGUF** 按本分支的
ROCm 优化类型混合量化（直接调 `libggml-base.so` 的 `*_ref` 量化函数，无需 llama-quantize）：

```bash
cd /opt/LLM/working/jgbrblmd/mx-llama.cpp   # 需 build/bin/libggml-base.so
/opt/venv/common/bin/python tools/convert-qwen35moe-rocmfp4.py \
  /models/Qwen3.5-...-bf16.gguf /models/Qwen3.5-...-ROCMFP4.gguf \
  --output rocmfp4 -t 8
```

| `--output` | file_type | 配方 |
|---|---|---|
| `rocmfp4`（默认） | 100 | 默认 `Q4_0_ROCMFP4`；`output`/token_embd → `Q6_K`；`attn_qkv`/`attn_v` → `Q5_K`；`ffn_down` 前 2/3 层 `Q6_K` 其余 `Q5_K`；`ffn_gate` 关键层 `Q5_K` |
| `strix_leon` | 106 | 默认 `Q4_0_ROCMFP4_FAST`；`output`/token_embd → `Q5_K`；`attn` → `Q4_0_ROCMFP4`；`ffn_*` → `Q4_0_ROCMFP4_FAST` |
| `rocmfpx` | 111 | 默认 `Q8_0_ROCMFPX`；`attn_qkv`/首末层 `ffn_down`/`ffn_up` → 普通 `Q8_0`；其余 `Q8_0_ROCMFPX` |

- 层数从 `qwen35moe.block_count` / `qwen35.block_count` 读取（缺失时按 `blk.*` 张量名推断）
- 1D 张量、`*_norm.weight`、`ssm_conv1d`、bias 等强制保持 F32
- 产物直接喂给本镜像推理，用法同「1. 单机单卡」；`-t` 为量化线程数

## 重新构建

```bash
# 1. 编译（输出到仓库根 build/）
bash mx-llama.cpp/jgbrblmd/build-rocm716.sh

# 2. 重新生成 staging（rocm/ + llama/ 会被重建，需本机装有 ROCm 7.16 SDK @ /opt/rocm-10.1.0）
bash mx-llama.cpp/jgbrblmd/docker/stage.sh

# 3. 构建镜像
cd mx-llama.cpp/jgbrblmd/docker && docker build \
  --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) \
  -t jgbrblmd123/mx-llama.cpp:gfx906-rocm716 .
```

目录布局（仓库内 `mx-llama.cpp/jgbrblmd/`）：

```
jgbrblmd/
├── build-rocm716.sh          # 编译脚本（ROCm 7.16 SDK @ /opt/rocm-10.1.0，可用 ROC 覆盖）
└── docker/
    ├── Dockerfile
    ├── stage.sh               # ldd 闭包 + 拷贝，重新生成 rocm/ llama/
    ├── rocm/                  # ROCm 7.16 运行时（~590MB）
    ├── llama/                 # 可执行文件 + so（~104MB）
    └── RELEASE.md             # 本文件
```

> `rocm/`、`llama/` 是构建镜像用的二进制 staging 产物（共 ~690MB），
> 提交 git 前建议加入 `.gitignore`（保留 Dockerfile/stage.sh/RELEASE.md 即可，staging 可随时用 stage.sh 重新生成）；若仓库走 LFS 或内部制品库则可直接提交。

## 限制

- **runtime-only**：镜像内无 `hipcc`/头文件，不能编译，只能跑
- TurboQuant 要求全层 GPU + Flash Attention；2-GPU layer-split 组合未验证
- `ct_int4` / turbo 系列在多卡 TP 下的行为以单卡/TP 实测为准
