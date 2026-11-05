"""End-to-end pipeline test with a synthetic (download-free) corpus.

Builds a fake corpus (random-weight networks + template captions), fakes the
text encoder, trains the diffusion model for a handful of steps, and runs the
generate path from prompt to GIF. Validates mechanics, not model quality.
"""

import hashlib
import json

import numpy as np
import torch

from dnn.config import NUM_PARAMS, get_preset
from dnn.controller import ParamStats, flat_to_canvas, mask_flat
from dnn.corpus.embed_captions import embed_captions
from dnn.corpus.store import Corpus, save_weights
from dnn.diffusion.sampler import Diffusion
from dnn.diffusion.unet import UNet
from dnn.render import render_gif, render_weights
from dnn.tasks import get_task


class FakeEncoder:
    """Duck-typed TextEncoder: deterministic hash-based pseudo-embeddings."""

    def encode(self, captions, max_len=64):
        outs, masks = [], []
        for c in captions:
            g = torch.Generator().manual_seed(int(hashlib.sha256(c.encode()).hexdigest()[:8], 16))
            n_tokens = max(2, min(max_len, len(c.split()) + 2))
            outs.append(torch.randn(n_tokens, 384, generator=g))
            masks.append(torch.ones(n_tokens))
        L = max(m.shape[0] for m in outs)
        emb = torch.stack([torch.cat([e, torch.zeros(L - e.shape[0], 384)]) for e in outs])
        mask = torch.stack([torch.cat([m, torch.zeros(L - m.shape[0])]) for m in masks])
        return emb, mask


def _build_fake_corpus(root, per_task=12):
    torch.manual_seed(0)
    tasks = ["seek", "draw_circle", "gait"]
    weights, metas = [], []
    for task in tasks:
        # Weights with realistic per-tensor scales (std ~ 0.1-0.25).
        w = torch.randn(per_task, NUM_PARAMS) * 0.15
        for i in range(per_task):
            weights.append(w[i])
            metas.append(
                {
                    "id": len(metas),
                    "task": task,
                    "caption": f"A tiny MLP controller for the {task} family, sample {i}.",
                    "score": 0.5 + 0.1 * (i % 4),
                    "success": True,
                    "split": "val" if i >= per_task - 2 else "train",
                    "seed": i,
                }
            )
    weights = torch.stack(weights)
    stats = ParamStats.from_flat(weights)
    save_weights(root, weights, metas, stats)
    embed_captions(root, encoder=FakeEncoder())
    return corpus_ok(root)


def corpus_ok(root):
    c = Corpus(root)
    b = c.sample_batch(4)
    assert b["canvas"].shape == (4, 6, 32, 32)
    assert b["emb"].shape[2] == 384
    assert torch.isfinite(b["emb"]).all()
    # Normalized canvases: valid entries within [-1, 1], padding zero.
    m = flat_to_canvas(mask_flat().unsqueeze(0)).bool()[0]
    assert b["canvas"][:, m.bool() == False].abs().sum() == 0
    assert b["canvas"].abs().max() <= 1.0 + 1e-6
    return c


def test_fake_corpus(tmp_path):
    _build_fake_corpus(tmp_path)


def test_train_and_generate_smoke(tmp_path):
    import shutil

    run_dir = tmp_path / "runs" / "smoke"
    run_dir.mkdir(parents=True)
    from dnn.train_diffusion import CKPT_FILE, main as train_main  # noqa: F401

    # Use the real training loop via its pieces (CLI main uses argparse).
    import sys

    sys.argv = [
        "train",
        "--config", "smoke",
        "--corpus", str(tmp_path),
        "--run", str(run_dir),
        "--steps", "25",
    ]
    import contextlib
    import io as _io

    with contextlib.redirect_stdout(_io.StringIO()):
        train_main()

    ckpt = torch.load(run_dir / CKPT_FILE, map_location="cpu", weights_only=False)
    assert "ema" in ckpt and ckpt["step"] >= 24
    assert "stats" in ckpt

    # Generate path: prompt -> canvas -> controller -> rollout -> artifacts.
    from dnn.generate import canvas_to_controller, generate_canvas

    canvas, stats, info = generate_canvas(
        "A tiny MLP controller for the seek family, sample 3.",
        run_dir / CKPT_FILE,
        guidance=2.0,
        steps=4,
        seed=1,
        encoder=FakeEncoder(),
    )
    assert canvas.shape == (1, 6, 32, 32)
    net = canvas_to_controller(canvas, stats)
    task = get_task("seek")
    states = task.rollout(net, 1, 999)
    score = float(task.score(states)[0])
    assert np.isfinite(score)

    out = tmp_path / "demo"
    out.mkdir()
    render_weights(flat_to_canvas(net.get_flat().unsqueeze(0))[0], out / "weights.png")
    render_gif("seek", task, states, out / "behavior.gif")
    assert (out / "weights.png").stat().st_size > 1000
    assert (out / "behavior.gif").stat().st_size > 5000
