#!/usr/bin/env bash
# 重新生成 docker 构建的 staging 内容（rocm/ + llama/）。
# 前提：已按 ../docs/build-rocm716.sh 编译完成 build/ 目录。
# 用法：bash stage.sh
set -euo pipefail
STG="$(cd "$(dirname "$0")" && pwd)"          # .../mx-llama.cpp/jgbrblmd/docker
REPO="$(cd "$STG/../.." && pwd)"              # mx-llama.cpp 仓库根
ROC=${ROC:-/opt/rocm-10.1.0}                  # ROCm 7.16 SDK（rocm 10.1.0 命名）
BUILD=$REPO/build
export ROC BUILD STG

/opt/venv/common/bin/python - <<'EOF'
import subprocess, os, shutil
ROC=os.environ['ROC']
BUILD=os.path.join(os.environ['BUILD'], 'bin')
STG=os.environ['STG']

# 1. 传递闭包：所有 llama 二进制 + 本地 so 对 ROCm SDK 的 ldd 依赖
env=dict(os.environ, LD_LIBRARY_PATH=ROC+'/lib')
files=[os.path.join(BUILD,f) for f in os.listdir(BUILD)
       if f.startswith('llama-') or f.endswith(('.so','.so.0','.so.1'))]
deps=set(); queue=list(files); seen=set(files)
while queue:
    f=queue.pop()
    out=subprocess.run(['ldd',f],env=env,capture_output=True,text=True).stdout
    for line in out.splitlines():
        if ' => ' not in line: continue
        p=line.split(' => ')[1].split()[0]
        if p.startswith(ROC) and '.so' in p:
            real=os.path.realpath(p)
            if real not in deps and real not in seen:
                deps.add(real); queue.append(real); seen.add(real)
print(f'ROCm lib closure: {len(deps)} libs')

# 2. 清理并重建 staging
shutil.rmtree(f'{STG}/rocm', ignore_errors=True)
shutil.rmtree(f'{STG}/llama', ignore_errors=True)
for p in sorted(deps):
    rel=os.path.relpath(p, ROC)
    dst=f'{STG}/rocm/{rel}'
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(p, dst)

# SONAME 符号链接（target 在闭包内才复制，保持相对链接）
for d in ['lib','lib/llvm/lib','lib/rocm_sysdeps/lib']:
    src_dir=f'{ROC}/{d}'
    for name in os.listdir(src_dir):
        sp=os.path.join(src_dir,name)
        if os.path.islink(sp) and os.path.realpath(sp) in deps:
            dst=f'{STG}/rocm/{d}/{name}'
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.symlink(os.readlink(sp), dst)

# 3. rocBLAS 预编译 kernel（gfx906）+ rccl share
shutil.copytree(f'{ROC}/lib/rocblas/library', f'{STG}/rocm/lib/rocblas/library')
rccl=f'{ROC}/share/rccl'
if os.path.isdir(rccl):
    shutil.copytree(rccl, f'{STG}/rocm/share/rccl')

# 4. llama 二进制与 so
os.makedirs(f'{STG}/llama/bin', exist_ok=True)
os.makedirs(f'{STG}/llama/lib', exist_ok=True)
for f in os.listdir(BUILD):
    sp=os.path.join(BUILD,f)
    if f.startswith('llama-') and not (f.endswith('.so') or '.so.' in f):
        if not os.path.islink(sp):
            shutil.copy2(sp, f'{STG}/llama/bin/{f}')
    elif f.endswith('.so') or '.so.' in f:
        if os.path.islink(sp) and os.path.realpath(sp).startswith(BUILD):
            os.symlink(os.readlink(sp), f'{STG}/llama/lib/{f}')
        elif not os.path.islink(sp):
            shutil.copy2(sp, f'{STG}/llama/lib/{f}')
print('staging done ->', STG)
EOF

echo
echo "构建镜像："
echo "  cd $STG && docker build --build-arg GIT_COMMIT=\$(git -C $REPO rev-parse --short HEAD) -t jgbrblmd/mx-llama.cpp:gfx906-rocm716 ."
