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
from ..tasks.draw import HELD_OUT_SHAPES
from .captions import build_caption
from .store import save_weights

# Tasks registered for evaluation only; never part of "--tasks all".
EVAL_ONLY_TASKS = tuple(f"draw_{s}" for s in HELD_OUT_SHAPES)

SEED_OFFSET = 1_000_000
EVAL_SEED_OFFSET = 10_000

# Per-task overrides: tasks whose dynamics need longer BPTT training.
TRAIN_STEPS_OVERRIDES = {"bounce": 600, "evade": 500}


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
        steps = TRAIN_STEPS_OVERRIDES.get(task_name, train_steps)
        flat, scores, success = trainer.train_batch(seed, lr, steps)
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


# -- incremental per-task cache (crash-resilience) ---------------------------
def _partial_dir(out: Path) -> Path:
    d = out.parent / (out.name + "_partial")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_partial(out: Path, name: str, weights: torch.Tensor, metas: list[dict]) -> None:
    d = _partial_dir(out)
    save_weights(d / name, weights, metas, ParamStats.from_flat(weights))
    print(f"[cache] task '{name}' saved to {d / name}", flush=True)


def _load_partial(out: Path, name: str) -> tuple[torch.Tensor, list[dict]] | None:
    d = _partial_dir(out) / name
    if (d / "weights.safetensors").exists():
        from safetensors.torch import load_file

        w = load_file(d / "weights.safetensors")["weights"]
        m = [json.loads(l) for l in open(d / "meta.jsonl")]
        return w, m
    return None


def _renumber(all_metas: list[dict], metas: list[dict]) -> list[dict]:
    offset = len(all_metas)
    renumbered = [{**m, "id": m["id"] + offset} for m in metas]
    return renumbered


def safe_worker(payload: dict):
    """Pool-map wrapper: isolates per-task failures (must be picklable)."""
    try:
        return _worker(payload), None
    except Exception as e:
        return (payload["task"], torch.zeros(0), []), f"{payload['task']}: {e}"


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
    ap.add_argument("--no-cache", action="store_true", help="ignore per-task partial cache")
    args = ap.parse_args()

    names = list(TASKS) if args.tasks == "all" else [t.strip() for t in args.tasks.split(",")]
    skipped_eval_only = [n for n in names if n in EVAL_ONLY_TASKS]
    if skipped_eval_only:
        print(f"Skipping held-out eval-only tasks (never trained): {', '.join(skipped_eval_only)}", flush=True)
        names = [n for n in names if n not in EVAL_ONLY_TASKS]
    for n in names:
        get_task(n)  # validate

    out = Path(args.out)
    # Skip tasks already cached from a previous (crashed) run.
    pending, cached = [], []
    all_weights: list[torch.Tensor] = []
    all_metas: list[dict] = []
    for n in names:
        part = _load_partial(out, n) if not args.no_cache else None
        if part is not None:
            w, m = part
            all_weights.append(w)
            all_metas.extend(_renumber(all_metas, m))
            cached.append(n)
        else:
            pending.append(n)
    if cached:
        print(f"[cache] resuming: {len(cached)} tasks cached ({', '.join(cached)}), {len(pending)} to generate", flush=True)

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
        for n in pending
    ]

    failed: list[str] = []
    if not payloads:
        pass  # everything cached
    elif args.workers <= 1 or len(pending) == 1:
        for p in payloads:
            try:
                _, w, m = _worker(p)
                _save_partial(out, p["task"], w, m)
                all_weights.append(w)
                all_metas.extend(_renumber(all_metas, m))
            except Exception as e:  # a task that can't hit its quota must not sink the run
                failed.append(p["task"])
                print(f"[pool] task '{p['task']}' FAILED: {e}", flush=True)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(args.workers, len(pending))) as pool:
            for (name, w, m), err in pool.imap_unordered(safe_worker, payloads):
                if err:
                    failed.append(name)
                    print(f"[pool] task '{err}' FAILED", flush=True)
                    continue
                _save_partial(out, name, w, m)
                print(f"[pool] task '{name}' returned {w.shape[0]} networks", flush=True)
                all_weights.append(w)
                all_metas.extend(_renumber(all_metas, m))

    if not all_weights:
        raise SystemExit("All tasks failed; no corpus written. See logs above.")
    weights = torch.cat(all_weights)
    stats = ParamStats.from_flat(weights)
    root = Path(args.out)
    save_weights(root, weights, all_metas, stats)
    print(f"Corpus saved to {root}: {weights.shape[0]} networks, {len(set(m['task'] for m in all_metas))} tasks")
    if failed:
        print(f"FAILED TASKS (rerun separately): {', '.join(failed)}")

    weights = torch.cat(all_weights)
    stats = ParamStats.from_flat(weights)
    root = Path(args.out)
    save_weights(root, weights, all_metas, stats)
    print(f"Corpus saved to {root}: {weights.shape[0]} networks, {len(set(m['task'] for m in all_metas))} tasks")


if __name__ == "__main__":
    main()
