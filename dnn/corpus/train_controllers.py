"""Mass-produce (network, caption) training pairs.

For every task, trains batches of canonical controllers in parallel through
the differentiable dynamics (a fresh environment seed every optimizer step,
so networks generalize across task instances rather than memorizing one
episode), evaluates each network on a held-out episode, keeps the successful
ones, and writes raw weights + captions to the corpus.

Performance: the per-step rollout+loss+backward is traced once per task with
torch.compile (~10x on CPU); tasks are generated in parallel worker processes.

Usage:
    python -m dnn.corpus.train_controllers --tasks all --per-task 600 --out data/corpus
    python -m dnn.corpus.train_controllers --tasks seek --per-task 64 --no-compile
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import torch

from ..controller import BatchedController, ParamStats
from ..tasks import TASKS, get_task
from .captions import build_caption
from .store import save_weights

SEED_OFFSET = 1_000_000
EVAL_SEED_OFFSET = 10_000


def _batch_seed(task_name: str, batch_id: int, val: bool, salt: str = "") -> int:
    h = hashlib.sha256(f"{task_name}:{batch_id}:{val}:{salt}".encode()).hexdigest()
    return int(h[:8], 16)


class TaskTrainer:
    """Persistent batch trainer for one task: compiled rollout+loss, eager Adam.

    The compiled graph is built once per task (per process); every batch
    re-initializes the controller parameters in place and resets the optimizer.
    """

    def __init__(self, task, batch: int, use_compile: bool = True) -> None:
        self.task = task
        self.batch = batch
        self.net = BatchedController(batch, seed=0)

        def step(state: dict, horizon: int) -> torch.Tensor:
            states = [state]
            for _ in range(horizon):
                obs = task.observe(state)
                action = self.net(obs)
                state = task.dynamics(state, action)
                states.append(state)
            loss = task.train_loss(states)
            loss.backward()
            return loss.detach()

        self._step = torch.compile(step, dynamic=False) if use_compile else step

    def train_batch(self, seed: int, lr: float, steps: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.net.reinit_(seed)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        gen = torch.Generator().manual_seed(seed)
        full = self.task.spec.steps
        for step in range(steps):
            # Horizon ramp: short rollouts early (stabilizes BPTT through the
            # dynamics), full episodes once the policy is roughly competent.
            frac = step / max(1, steps)
            horizon = 30 if frac < 0.3 else (60 if frac < 0.6 else full)
            state = self.task.reset(self.batch, gen)  # new env instance per step
            opt.zero_grad(set_to_none=True)
            self._step(state, horizon)
            opt.step()
        with torch.no_grad():
            eval_state = self.task.rollout(self.net, self.batch, seed=(seed + EVAL_SEED_OFFSET) & 0x7FFF_FFFF)
            scores = self.task.score(eval_state)
            success = self.task.success(eval_state)
            flat = self.net.get_flat()
        return flat, scores, success


def generate_task(
    task_name: str,
    per_task: int,
    batch: int,
    train_steps: int,
    val_frac: float,
    use_compile: bool = True,
    log: bool = True,
) -> tuple[torch.Tensor, list[dict]]:
    """Generate all networks for one task. Returns (weights, metas)."""
    task = get_task(task_name)
    val_quota = max(24, round(per_task * val_frac))
    train_quota = per_task
    quota = train_quota + val_quota
    max_batches = (quota // batch + 4) * 4  # generous retry budget

    trainer = TaskTrainer(task, batch, use_compile=use_compile)
    kept: list[tuple[torch.Tensor, float, bool, str]] = []  # (flat, score, success, split)
    batch_id = 0
    t0 = time.time()
    while len(kept) < quota and batch_id < max_batches:
        split = "train" if len(kept) < train_quota else "val"
        val = split == "val"
        seed = _batch_seed(task_name, batch_id, val)
        lr_rng = torch.Generator().manual_seed(seed)
        # log-uniform around the sweet spot found by pilots (~3e-3, 250 steps)
        lr = 10 ** torch.empty(1).uniform_(-2.7, -2.4, generator=lr_rng).item()
        flat, scores, success = trainer.train_batch(seed, lr, train_steps)
        for i in range(batch):
            if len(kept) >= quota:
                break
            if bool(success[i]):
                kept.append((flat[i].clone(), float(scores[i]), True, split))
        batch_id += 1
        if log and batch_id % 2 == 0:
            print(
                f"  [{task_name}] batch {batch_id}: kept {len(kept)}/{quota} "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )
    if len(kept) < quota:
        raise RuntimeError(
            f"Task '{task_name}': only {len(kept)}/{quota} successful networks after "
            f"{batch_id} batches ({time.time() - t0:.0f}s). Lower per-task, raise "
            f"--train-steps, or check the task threshold."
        )

    all_scores = torch.tensor([k[1] for k in kept])
    order = all_scores.argsort().argsort().float() / (len(kept) - 1)  # rank in [0, 1]

    weights = torch.stack([k[0] for k in kept])
    metas = []
    for i, (flat, score, success, split) in enumerate(kept):
        caption = build_caption(
            task_name, task.spec.io_desc, float(order[i]), score, seed=_batch_seed(task_name, i, False, salt="caption")
        )
        metas.append(
            {
                "id": i,
                "task": task_name,
                "caption": caption,
                "score": round(score, 4),
                "success": bool(success),
                "split": split,
                "seed": _batch_seed(task_name, i, False, salt="meta"),
            }
        )
    print(
        f"[{task_name}] done: {len(kept)} networks "
        f"({len([m for m in metas if m['split'] == 'val'])} val) in {time.time() - t0:.0f}s; "
        f"score mean {all_scores.mean():.3f}",
        flush=True,
    )
    return weights, metas


def _worker(payload: dict) -> tuple[str, torch.Tensor, list[dict]]:
    name, per_task, batch, steps, use_compile = (
        payload["task"], payload["per_task"], payload["batch"], payload["train_steps"], payload["compile"],
    )
    torch.set_num_threads(payload["threads"])
    weights, metas = generate_task(name, per_task, batch, steps, payload["val_frac"], use_compile=use_compile)
    return name, weights, metas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default="all", help="'all' or comma-separated task names")
    ap.add_argument("--per-task", type=int, default=600)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--train-steps", type=int, default=400)
    ap.add_argument("--val-frac", type=float, default=0.08)
    ap.add_argument("--out", default="data/corpus")
    ap.add_argument("--workers", type=int, default=3, help="parallel task workers")
    ap.add_argument("--threads-per-worker", type=int, default=2)
    ap.add_argument("--no-compile", action="store_true", help="disable torch.compile (slower, debug)")
    args = ap.parse_args()

    names = list(TASKS) if args.tasks == "all" else [t.strip() for t in args.tasks.split(",")]
    for n in names:
        get_task(n)  # validate

    payloads = [
        {
            "task": n,
            "per_task": args.per_task,
            "batch": args.batch,
            "train_steps": args.train_steps,
            "val_frac": args.val_frac,
            "compile": not args.no_compile,
            "threads": args.threads_per_worker,
        }
        for n in names
    ]

    all_weights: list[torch.Tensor] = []
    all_metas: list[dict] = []
    if args.workers <= 1 or len(names) == 1:
        for p in payloads:
            _, w, m = _worker(p)
            all_weights.append(w)
            all_metas.extend(m)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(args.workers, len(names))) as pool:
            for name, w, m in pool.imap_unordered(_worker, payloads):
                print(f"[pool] task '{name}' returned {w.shape[0]} networks", flush=True)
                all_weights.append(w)
                all_metas.extend(m)

    weights = torch.cat(all_weights)
    stats = ParamStats.from_flat(weights)
    root = Path(args.out)
    save_weights(root, weights, all_metas, stats)
    print(f"Corpus saved to {root}: {weights.shape[0]} networks, {len(set(m['task'] for m in all_metas))} tasks")


if __name__ == "__main__":
    main()
