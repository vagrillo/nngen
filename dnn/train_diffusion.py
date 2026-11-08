"""Train the text-to-network diffusion model on a corpus.

Usage:
    python -m dnn.train_diffusion --config small --corpus data/corpus --run runs/small
    python -m dnn.train_diffusion --config small --run runs/small --resume
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .config import DiffusionConfig, TrainConfig, get_preset
from .corpus.store import Corpus
from .diffusion.sampler import Diffusion, EMA
from .diffusion.unet import UNet

CKPT_FILE = "ckpt.pt"
LOG_FILE = "log.jsonl"


def save_checkpoint(
    run_dir: Path,
    model: UNet,
    ema: EMA,
    opt: torch.optim.Optimizer,
    step: int,
    model_cfg: DiffusionConfig,
    train_cfg: TrainConfig,
    stats: dict,
    corpus: str,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.shadow,
            "opt": opt.state_dict(),
            "step": step,
            "model_cfg": asdict(model_cfg),
            "train_cfg": asdict(train_cfg),
            "stats": stats,
            "corpus": str(corpus),
        },
        run_dir / CKPT_FILE,
    )


def load_checkpoint(path: str | Path) -> dict:
    """Load a checkpoint from a run directory, a ckpt.pt file, or a slim
    release .safetensors (EMA weights + metadata, see scripts/export_release.py)."""
    p = Path(path)
    if p.is_dir():
        p = p / CKPT_FILE
    if p.suffix == ".safetensors":
        from safetensors.torch import load_file

        raw = load_file(str(p))
        meta = {}
        try:  # safetensors exposes file metadata via safe_open
            from safetensors import safe_open

            with safe_open(str(p), framework="pt") as f:
                meta = f.metadata() or {}
        except Exception:
            meta = {}
        return {
            "ema": {k: v.float() for k, v in raw.items()},
            "step": int(meta.get("step", -1)),
            "model_cfg": json.loads(meta.get("model_cfg", "{}")),
            "stats": json.loads(meta.get("stats", "{}")),
            "corpus": meta.get("corpus", ""),
        }
    return torch.load(p, map_location="cpu", weights_only=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="small", help="preset: smoke | small | base")
    ap.add_argument("--corpus", default="data/corpus")
    ap.add_argument("--run", default=None, help="run directory (default runs/<config>)")
    ap.add_argument("--steps", type=int, default=None, help="override preset training steps")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    preset = get_preset(args.config)
    model_cfg = preset.model
    train_cfg = preset.train
    if args.steps is not None:
        train_cfg = TrainConfig(**{**asdict(train_cfg), "steps": args.steps})

    run_dir = Path(args.run) if args.run else Path("runs") / args.config
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(min(8, os.cpu_count() or 1))
    corpus = Corpus(args.corpus)
    print(f"Corpus: {len(corpus)} samples ({len(corpus.train_idx)} train / {len(corpus.val_idx)} val) from {args.corpus}")

    model = UNet(model_cfg)
    diff = Diffusion(model_cfg)
    ema = EMA(model, train_cfg.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    start = 0
    best_val = float("inf")
    if args.resume and (run_dir / CKPT_FILE).exists():
        ckpt = load_checkpoint(run_dir)
        model.load_state_dict(ckpt["model"])
        ema.shadow = {k: v.clone() for k, v in ckpt["ema"].items()}
        opt.load_state_dict(ckpt["opt"])
        start = ckpt["step"] + 1
        print(f"Resumed from step {ckpt['step']}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNet parameters: {n_params/1e6:.2f}M | steps: {start}..{train_cfg.steps} | batch {train_cfg.batch_size}")

    gen = torch.Generator().manual_seed(12345)
    log_file = open(run_dir / LOG_FILE, "a")
    history: list[dict] = []

    def lr_at(step: int) -> float:
        if step < train_cfg.warmup:
            return train_cfg.lr * step / max(1, train_cfg.warmup)
        p = (step - train_cfg.warmup) / max(1, train_cfg.steps - train_cfg.warmup)
        return train_cfg.lr * 0.5 * (1 + __import__("math").cos(3.14159 * min(1.0, p)))

    t0 = time.time()
    running = 0.0
    for step in range(start, train_cfg.steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step)

        batch = corpus.sample_batch(train_cfg.batch_size, generator=gen)
        loss = diff.training_loss(model, batch["canvas"], batch["emb"], batch["emb_mask"], corpus.null_embedding, generator=gen)
        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update(model)
        running += loss.item()

        if (step + 1) % train_cfg.log_every == 0:
            entry = {
                "step": step + 1,
                "train_loss": running / train_cfg.log_every,
                "lr": lr_at(step),
                "sec_per_step": (time.time() - t0) / train_cfg.log_every,
            }
            history.append(entry)
            log_file.write(json.dumps(entry) + "\n")
            log_file.flush()
            print(
                f"step {step+1}/{train_cfg.steps}  loss {entry['train_loss']:.4f}  "
                f"lr {entry['lr']:.2e}  {entry['sec_per_step']:.2f}s/step",
                flush=True,
            )
            running = 0.0
            t0 = time.time()

        if (step + 1) % train_cfg.val_every == 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(4):
                    vb = corpus.sample_batch(min(train_cfg.val_batch_size, len(corpus.val_idx)), split="val", generator=gen)
                    val_losses.append(
                        diff.training_loss(
                            model, vb["canvas"], vb["emb"], vb["emb_mask"], corpus.null_embedding,
                            generator=gen, cfg_dropout=0.0,
                        ).item()
                    )
            model.train()
            v = sum(val_losses) / len(val_losses)
            entry = {"step": step + 1, "val_loss": v}
            history.append(entry)
            log_file.write(json.dumps(entry) + "\n")
            log_file.flush()
            print(f"step {step+1}  val_loss {v:.4f}", flush=True)
            # best-val checkpointing: small corpora overfit fast, so keep the
            # best generalizing weights, not the last ones
            if v < best_val:
                best_val = v
                save_checkpoint(
                    run_dir, model, ema, opt, step, model_cfg, train_cfg, corpus.stats.to_json(), args.corpus
                )
                (run_dir / "BEST").write_text(f"step {step+1} val_loss {v:.4f}\n")
                print(f"best-val checkpoint (step {step+1}, {v:.4f}) -> {run_dir / CKPT_FILE}", flush=True)

        if (step + 1) % train_cfg.ckpt_every == 0 or (step + 1) == train_cfg.steps:
            save_checkpoint(
                run_dir, model, ema, opt, step, model_cfg, train_cfg, corpus.stats.to_json(), args.corpus
            )
            print(f"checkpoint saved -> {run_dir / CKPT_FILE}", flush=True)

    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f)
    print("Training complete.")


if __name__ == "__main__":
    main()
