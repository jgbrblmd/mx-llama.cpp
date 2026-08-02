#!/usr/bin/env python3
"""Fix official ModelOpt NVFP4 GGUF: multiply the per-expert scale tensors by K.

The official pack stores expert weights normalized by a global constant K
(~18479, fitted vs the BF16 source; unsloth MXFP4 builds are NOT normalized and
run fine). llama.cpp's MoE graph multiplies mul_mat_id outputs by these scale
tensors, so scaling the tensors by K folds the normalization back into the
weights where per-expert variation prevents layer norms from absorbing it.

Usage: fix-official-nvfp4-scale.py <input.gguf> <output.gguf>
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gguf-py'))
from gguf import GGUFReader

K = 18479.0  # global constant fitted vs BF16 source (least squares, spread < 1%)

def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    src_path, dst_path = sys.argv[1], sys.argv[2]
    r = GGUFReader(src_path, 'r')

    n_expert = 0
    with open(src_path, 'rb') as fin, open(dst_path, 'wb') as fout:
        first = min(t.data_offset for t in r.tensors)
        fin.seek(0)
        fout.write(fin.read(first))
        fout.seek(first)
        for t in r.tensors:
            fin.seek(t.data_offset)
            nbytes = t.n_bytes
            if t.name.endswith('.scale') and '_exps.' in t.name:
                data = np.frombuffer(fin.read(nbytes), dtype=np.float32).copy()
                data *= K
                fout.write(data.tobytes())
                n_expert += 1
            else:
                fout.write(fin.read(nbytes))
        fin.seek(first)
        last_end = max(t.data_offset + t.n_bytes for t in r.tensors)
        fin.seek(last_end)
        fout.write(fin.read())

    print(f'scaled {n_expert} scale tensors by K={K} -> {dst_path}')
    return 0

if __name__ == '__main__':
    sys.exit(main())
