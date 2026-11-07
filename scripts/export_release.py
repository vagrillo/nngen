"""Export a slim, distribution-ready checkpoint of the trained generator.

Drops the optimizer state (the full training checkpoint is ~3x larger) and
stores everything inference needs: EMA weights + model config + corpus
normalization stats, as a safetensors file with JSON metadata.

Usage:
    python scripts/export_release.py --ckpt runs/draw_mvp/ckpt.pt \
        --out weights/nngen_draw_mvp.safetensors
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors.torch import save_file

from dnn.train_diffusion import load_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="runs/draw_mvp/ckpt.pt")
    ap.add_argument("--out", default="weights/nngen_draw_mvp.safetensors")
    args = ap.parse_args()

    ckpt = load_checkpoint(args.ckpt)
    ema = {k: v.to(torch.float32).contiguous() for k, v in ckpt["ema"].items()}
    meta = {
        "format": "nngen-slim-ckpt-v1",
        "model_cfg": json.dumps(ckpt["model_cfg"]),
        "stats": json.dumps(ckpt["stats"]),
        "step": str(ckpt.get("step", -1)),
        "corpus": ckpt.get("corpus", ""),
        "note": "EMA weights of the NNGen drawing generator; load with dnn.train_diffusion.load_checkpoint",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(ema, out, metadata=meta)
    print(f"exported {len(ema)} tensors ({out.stat().st_size/1e6:.1f} MB) -> {out}")


if __name__ == "__main__":
    main()
