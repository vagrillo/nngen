"""Pilot every task: one batch of controllers, report success rate.

Determines whether the default training recipe (lr, steps, thresholds) yields
enough successful networks per task before launching the full corpus run.

Usage:
    python scripts/pilot_tasks.py --tasks all --batch 32 --steps 350 --lr 2e-3
"""

from __future__ import annotations

import argparse
import time

import torch

from dnn.corpus.train_controllers import TaskTrainer
from dnn.tasks import TASKS


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=350)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    names = list(TASKS) if args.tasks == "all" else [t.strip() for t in args.tasks.split(",")]
    results = {}
    for name in names:
        task = get_task(name)
        trainer = TaskTrainer(task, args.batch, use_compile=not args.no_compile)
        t0 = time.time()
        flat, scores, success = trainer.train_batch(42, args.lr, args.steps)
        # Second batch with a different seed to confirm stability.
        flat2, scores2, success2 = trainer.train_batch(43, args.lr, args.steps)
        rate = (success.float().mean() + success2.float().mean()).item() / 2
        results[name] = (rate, float(scores.mean()), time.time() - t0)
        print(f"{name:14s} success {rate:5.0%}  score {scores.mean():.3f}/{scores.max():.3f}  ({results[name][2]:.0f}s)", flush=True)

    print("\nSummary (min success rate):", min(r[0] for r in results.values()))


if __name__ == "__main__":
    main()
