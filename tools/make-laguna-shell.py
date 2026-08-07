#!/usr/bin/env python3
"""Generate a metadata-only GGUF shell for Laguna from HF files.

Runs the official converter's metadata path (config.json + tokenizer.json ->
GGUF KV, including the laguna-specific hparams and per-layer head counts) but
skips tensor conversion entirely. The streaming quantizer
(convert-laguna-stream.py) then copies this metadata and writes ROCmFP4 tensor
data itself, so the 230GB BF16 intermediate is never materialized and nothing
depends on the ModelOpt GGUF (which is slated for deletion).

Usage: python make-laguna-shell.py <src-dir> <out-shell.gguf>
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'gguf-py'))  # repo gguf-py (has LAGUNA arch), not the pip install
sys.path.insert(0, str(ROOT))

import gguf
from conversion import (
    ModelBase,
    ModelType,
    get_model_architecture,
    get_model_class,
    logger,
)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    src = Path(sys.argv[1])
    out = Path(sys.argv[2])

    with torch.inference_mode():
        output_type = gguf.LlamaFileType.MOSTLY_BF16
        hparams = ModelBase.load_hparams(src, is_mistral_format=False)
        model_architecture = get_model_architecture(hparams, ModelType.TEXT)
        model_class = get_model_class(model_architecture, mmproj=False)
        logger.info(f"Model architecture: {model_architecture}")

        model_instance = model_class(src, output_type, out)
        # metadata + vocab only: prepare_metadata adds all KV entries; skip
        # prepare_tensors / write_tensors so no tensor data is emitted
        model_instance.prepare_metadata(vocab_only=False)
        model_instance.gguf_writer.write_header_to_file(path=out)
        model_instance.gguf_writer.write_kv_data_to_file()
        model_instance.gguf_writer.close()

    print(f'shell written: {out} ({out.stat().st_size} bytes)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
