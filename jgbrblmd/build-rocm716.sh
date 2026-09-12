# 与 build.sh 相同，但使用从 zenth815/rocm10-gfx906:sdk 提取的 ROCm 7.16
# （/opt/rocm-10.1.0，可用环境变量 ROC 覆盖）编译，输出到仓库根下 build/。
# 脚本位于 jgbrblmd/ 目录内，任意工作目录执行均可。
cd "$(dirname "$0")/.." && ROC=${ROC:-/opt/rocm-10.1.0}

#cmake -B build-rocm716 \

cmake -B build \
  -DGGML_HIP=ON \
  -DGGML_HIP_GRAPHS=ON \
  -DGGML_HIP_RCCL=ON \
  -DLLAMA_OPENSSL=ON \
  -DAMDGPU_TARGETS=gfx906 \
  -DCMAKE_BUILD_TYPE=Release \
  -DHIP_COMPILER=clang \
  -DCMAKE_HIP_COMPILER=$ROC/lib/llvm/bin/clang++ \
  -DCMAKE_HIP_COMPILER_AR=$ROC/lib/llvm/bin/llvm-ar \
  -DCMAKE_HIP_COMPILER_RANLIB=$ROC/lib/llvm/bin/llvm-ranlib \
  -DCMAKE_PREFIX_PATH=$ROC \
  -DCMAKE_CXX_FLAGS="-O3 -Wno-unused-command-line-argument" \
  -DCMAKE_HIP_FLAGS="-mllvm -amdgpu-sched-strategy=max-ilp" \
  -DLLAMA_BUILD_TESTS=OFF

cmake --build build --config Release -j
