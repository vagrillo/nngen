"""Cache MiniLM embeddings for every corpus caption (run once after corpus
generation; requires one-time HuggingFace download).

Usage:
    python -m dnn.corpus.embed_captions --corpus data/corpus
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..text_encoder import MAX_LEN, TextEncoder, null_embedding
from .store import load_metas, save_embeddings


def embed_captions(root: Path, encoder: TextEncoder | None = None, chunk: int = 256) -> None:
    """Embed all corpus captions and the null prompt; writes cache files."""
    encoder = encoder or TextEncoder()
    captions = [m["caption"] for m in load_metas(root)]
    print(f"Encoding {len(captions)} captions...", flush=True)

    embs, masks = [], []
    for i in range(0, len(captions), chunk):
        emb, mask = encoder.encode(captions[i : i + chunk], max_len=MAX_LEN)
        embs.append(emb)
        masks.append(mask)
        if (i // chunk) % 5 == 0:
            print(f"  {min(i + chunk, len(captions))}/{len(captions)}", flush=True)

    # Uniform padded length across the corpus (per-batch lengths vary).
    L = max(e.shape[1] for e in embs)
    emb = torch.cat([torch.nn.functional.pad(e, (0, 0, 0, L - e.shape[1])) for e in embs])
    mask = torch.cat([torch.nn.functional.pad(m, (0, L - m.shape[1])) for m in masks])

    null_emb, null_mask = null_embedding(encoder, max_len=L)
    save_embeddings(root, emb, mask, null_emb, null_mask)
    print(f"Saved embeddings {tuple(emb.shape)} and null prompt to {root}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default="data/corpus")
    args = ap.parse_args()
    embed_captions(Path(args.corpus))


if __name__ == "__main__":
    main()
