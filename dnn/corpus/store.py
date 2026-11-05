"""Corpus storage format and loader.

A corpus directory contains:
    weights.safetensors   {"weights": fp32 (N, NUM_PARAMS)}     raw flat parameters
    meta.jsonl            one JSON per sample: id, task, caption, score, success, split
    stats.json            per-tensor mean/std used for canvas normalization
    emb.safetensors       {"emb": fp16 (N, L, TEXT_DIM)}        cached token embeddings
    emb_mask.safetensors  {"mask": fp16 (N, L)}                 token attention mask
    null.safetensors      {"emb": fp16 (L, TEXT_DIM), "mask": (L,)}   CFG null prompt
Embeddings are optional until `embed_captions` has run.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from ..config import TEXT_DIM
from ..controller import NUM_PARAMS, ParamStats, flat_to_canvas, normalize_canvas, canvas_mask

WEIGHTS_FILE = "weights.safetensors"
META_FILE = "meta.jsonl"
STATS_FILE = "stats.json"
EMB_FILE = "emb.safetensors"
EMB_MASK_FILE = "emb_mask.safetensors"
NULL_FILE = "null.safetensors"


def save_weights(root: Path, weights: torch.Tensor, metas: list[dict], stats: ParamStats) -> None:
    root.mkdir(parents=True, exist_ok=True)
    assert weights.shape == (len(metas), NUM_PARAMS)
    save_file({"weights": weights.contiguous()}, root / WEIGHTS_FILE)
    with open(root / META_FILE, "w") as f:
        for m in metas:
            f.write(json.dumps(m) + "\n")
    with open(root / STATS_FILE, "w") as f:
        json.dump(stats.to_json(), f)


def save_embeddings(root: Path, emb: torch.Tensor, mask: torch.Tensor, null_emb: torch.Tensor, null_mask: torch.Tensor) -> None:
    save_file({"emb": emb.to(torch.float16).contiguous()}, root / EMB_FILE)
    save_file({"mask": mask.to(torch.float16).contiguous()}, root / EMB_MASK_FILE)
    save_file({"emb": null_emb.to(torch.float16).contiguous(), "mask": null_mask.to(torch.float16).contiguous()}, root / NULL_FILE)


def load_metas(root: Path) -> list[dict]:
    with open(root / META_FILE) as f:
        return [json.loads(line) for line in f]


def load_stats(root: Path) -> ParamStats:
    with open(root / STATS_FILE) as f:
        return ParamStats.from_json(json.load(f))


def has_embeddings(root: Path) -> bool:
    return (root / EMB_FILE).exists()


class Corpus:
    """In-memory view of a corpus, ready for diffusion training and eval."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.weights = load_file(self.root / WEIGHTS_FILE)["weights"]  # (N, NUM_PARAMS)
        self.metas = load_metas(self.root)
        self.stats = load_stats(self.root)
        assert len(self.metas) == self.weights.shape[0]
        self.task_names = sorted({m["task"] for m in self.metas})
        self.train_idx = torch.tensor([i for i, m in enumerate(self.metas) if m["split"] == "train"])
        self.val_idx = torch.tensor([i for i, m in enumerate(self.metas) if m["split"] == "val"])
        self._emb = None
        self._emb_mask = None
        self._null = None

    # -- embeddings ---------------------------------------------------------
    def _load_embeddings(self) -> None:
        if self._emb is None:
            if not has_embeddings(self.root):
                raise RuntimeError(
                    f"Corpus at {self.root} has no cached embeddings; run `python -m dnn.corpus.embed_captions` first."
                )
            self._emb = load_file(self.root / EMB_FILE)["emb"].float()        # (N, L, D)
            self._emb_mask = load_file(self.root / EMB_MASK_FILE)["mask"].float()
            n = load_file(self.root / NULL_FILE)
            self._null = (n["emb"].float(), n["mask"].float())

    @property
    def null_embedding(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._load_embeddings()
        return self._null

    # -- batching ------------------------------------------------------------
    def sample_batch(self, batch: int, split: str = "train", generator: torch.Generator | None = None) -> dict:
        idx = self.train_idx if split == "train" else self.val_idx
        pick = torch.randint(0, len(idx), (batch,), generator=generator)
        return self.make_batch(idx[pick])

    def make_batch(self, idx: torch.Tensor) -> dict:
        """Normalized canvases + text conditioning for the given sample ids."""
        self._load_embeddings()
        flat = self.weights[idx]
        canvas = flat_to_canvas(flat)
        norm = normalize_canvas(canvas, self.stats)
        return {
            "canvas": norm,                                  # (B, C, 32, 32)
            "emb": self._emb[idx],                           # (B, L, D)
            "emb_mask": self._emb_mask[idx],                 # (B, L)
            "task": [self.metas[int(i)]["task"] for i in idx],
        }

    def canvas_mask(self) -> torch.Tensor:
        return canvas_mask()

    def __len__(self) -> int:
        return len(self.metas)
