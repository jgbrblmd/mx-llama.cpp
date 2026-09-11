#!/usr/bin/env python3
"""Add a low-precision draft lm_head copy to a GGUF.

The MTP draft graph picks its output projection from `blk.<mtp>.nextn.shared_head_head`
when the tensor is present, and falls back to `model.output` otherwise (see
src/models/qwen35.cpp graph_mtp). The TRUNK graph always uses `model.output`.

So writing a quantized copy of the target lm_head under the nextn name speeds up only
the draft steps - which re-read that 5120 x n_vocab matrix once per drafted token - while
leaving the target's final logits bit-identical. The draft is verified by the target, so
the only observable effect is a possible small shift in acceptance rate.

Usage:
    python scripts/add_mtp_draft_head.py in.gguf out.gguf [--qtype Q8_0]

The output is a full copy of the input with one tensor appended, so it needs as much
free disk as the input takes plus the size of the new tensor.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Prefer the in-tree gguf-py: the fork carries type registrations the installed one lacks.
_GGUF_PY = Path(__file__).resolve().parent.parent / "gguf-py"
if _GGUF_PY.is_dir():
    sys.path.insert(0, str(_GGUF_PY))

import gguf  # noqa: E402
from gguf import GGUFReader, GGUFWriter  # noqa: E402
from gguf.constants import GGMLQuantizationType as T  # noqa: E402
from gguf.quants import GGML_QUANT_SIZES, dequantize, quantize  # noqa: E402

logger = logging.getLogger("add-mtp-draft-head")

# Rows quantized per chunk. Q8_0 blocks run along ne0, so rows are independent; this only
# bounds peak RAM (a full float32 copy of a 5120 x 248320 lm_head is just under 5 GB).
ROW_CHUNK = 8192


def find_mtp_layer(reader: GGUFReader) -> int:
    """Index of the MTP block, taken from whichever blk.N.nextn.* tensor the file has."""
    for tensor in reader.tensors:
        m = re.match(r"blk\.(\d+)\.nextn\.", tensor.name)
        if m:
            return int(m.group(1))
    raise SystemExit("no blk.N.nextn.* tensor found - this GGUF has no MTP head")


def dequantize_rows(data: np.ndarray, qtype: T) -> np.ndarray:
    """Rows of a 2-D tensor, dequantized to float32."""
    if qtype == T.BF16:
        # gguf-py keeps BF16 as raw uint16; there is no native numpy bfloat16.
        return (data.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    if qtype == T.F32:
        return data.view(np.float32)
    if qtype == T.F16:
        return data.view(np.float16).astype(np.float32)
    return dequantize(data, qtype).astype(np.float32, copy=False)


def quantize_tensor(src: np.ndarray, src_type: T, qtype: T) -> np.ndarray:
    """Quantize a whole 2-D tensor, one row chunk at a time."""
    n_rows, n_bytes = src.shape
    ne0 = n_bytes // GGML_QUANT_SIZES[src_type][1]
    logger.info("quantizing %s: %d x %d (%s -> %s)", "src", n_rows, ne0,
                src_type.name, qtype.name)

    out: list[np.ndarray] = []
    for r0 in tqdm(range(0, n_rows, ROW_CHUNK), desc="Quantizing", unit="chunk"):
        chunk = dequantize_rows(src[r0:r0 + ROW_CHUNK], src_type).reshape(-1, ne0)
        out.append(quantize(chunk, qtype))
    return np.concatenate(out)


def copy_metadata(reader: GGUFReader, writer: GGUFWriter) -> None:
    for field in reader.fields.values():
        # GGUFWriter emits the architecture itself; GGUF.* are virtual bookkeeping fields.
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="source GGUF")
    ap.add_argument("output", type=Path, help="destination GGUF")
    ap.add_argument("--qtype", default="Q8_0", choices=["Q8_0", "Q6_K", "Q5_K", "Q4_K"],
                    help="type for the draft copy (default: Q8_0)")
    ap.add_argument("--src-tensor", default="output.weight", help="target lm_head tensor name")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output file")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.input.is_file():
        raise SystemExit(f"no such file: {args.input}")
    if args.output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {args.output} (pass --force)")
    qtype = T[args.qtype]

    reader = GGUFReader(args.input)

    src = next((t for t in reader.tensors if t.name == args.src_tensor), None)
    if src is None:
        raise SystemExit(f"{args.src_tensor} not found in {args.input}")

    mtp_il = find_mtp_layer(reader)
    dst_name = f"blk.{mtp_il}.nextn.shared_head_head.weight"
    if any(t.name == dst_name for t in reader.tensors):
        raise SystemExit(f"{dst_name} already present - nothing to do")

    qdata = quantize_tensor(src.data, src.tensor_type, qtype)
    logger.info("draft head: %s -> %.2f GiB (from %.2f GiB %s)",
                dst_name, qdata.nbytes / 2**30, src.n_bytes / 2**30, src.tensor_type.name)

    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    writer = GGUFWriter(args.output, arch=arch, endianess=reader.endianess)
    alignment = reader.fields.get(gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment.contents()

    copy_metadata(reader, writer)

    # Tensor infos must be declared in the same order their data is written below.
    for tensor in reader.tensors:
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype,
                               tensor.data.nbytes, tensor.tensor_type)
        if tensor.name == args.src_tensor:
            writer.add_tensor_info(dst_name, qdata.shape, qdata.dtype, qdata.nbytes,
                                   raw_dtype=qtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    total = sum(t.n_bytes for t in reader.tensors) + qdata.nbytes
    bar = tqdm(total=total, unit="byte", unit_scale=True, desc="Writing")
    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
        bar.update(tensor.n_bytes)
        if tensor.name == args.src_tensor:
            writer.write_tensor_data(qdata, tensor_endianess=reader.endianess)
            bar.update(qdata.nbytes)
    bar.close()
    writer.close()

    logger.info("wrote %s", args.output)


if __name__ == "__main__":
    main()
