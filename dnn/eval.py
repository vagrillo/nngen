"""Evaluate the trained generator: sample networks from validation captions
(never seen in training) and compare them against their "teacher" corpus
networks and a random baseline, on held-out environment seeds.

    python -m dnn.eval --ckpt runs/small/ckpt.pt --per-task 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import DiffusionConfig
from .controller import controller_from_flat
from .corpus.store import Corpus
from .diffusion.sampler import Diffusion
from .diffusion.unet import UNet
from .generate import canvas_to_controller, generate_canvas
from .tasks import get_task
from .text_encoder import TextEncoder

EVAL_SEED = 777_000


def eval_task(
    task_name: str,
    samples: list[dict],
    ckpt_path: Path,
    guidance: float,
    steps: int,
    encoder: TextEncoder,
    corpus: Corpus,
) -> dict:
    """Generate one net per caption; compare generated vs teacher vs random."""
    task = get_task(task_name)
    rows = []
    for i, meta in enumerate(samples):
        seed = 1000 + i
        # 1) Generated network from the caption.
        canvas, stats, _ = generate_canvas(meta["caption"], ckpt_path, guidance, steps, seed, encoder)
        gen_net = canvas_to_controller(canvas, stats)
        # 2) Teacher network: the corpus weights that caption described.
        teacher_net = controller_from_flat(corpus.weights[meta["id"]])
        # 3) Random baseline.
        torch.manual_seed(EVAL_SEED + seed)
        rand_net = controller_from_flat(torch.randn(1608) * 0.08)

        row = {"id": meta["id"], "score": meta["score"]}
        env_seed = EVAL_SEED + seed
        for label, net in (("generated", gen_net), ("teacher", teacher_net), ("random", rand_net)):
            states = task.rollout(net, 1, env_seed)
            row[f"{label}_score"] = round(float(task.score(states)[0]), 4)
            try:
                row[f"{label}_success"] = bool(task.success(states)[0])
            except NotImplementedError:
                row[f"{label}_success"] = None
        rows.append(row)

    def frac(key):
        vals = [r[key] for r in rows if r[key] is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    summary = {
        "task": task_name,
        "n": len(rows),
        "generated_success": frac("generated_success"),
        "teacher_success": frac("teacher_success"),
        "random_success": frac("random_success"),
        "generated_score_mean": round(sum(r["generated_score"] for r in rows) / len(rows), 3),
        "teacher_score_mean": round(sum(r["teacher_score"] for r in rows) / len(rows), 3),
    }
    return {"summary": summary, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="runs/small/ckpt.pt")
    ap.add_argument("--per-task", type=int, default=8)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    corpus = Corpus(ckpt.get("corpus", "data/corpus"))
    encoder = TextEncoder()

    # Val samples grouped by task.
    val_by_task: dict[str, list[dict]] = {}
    for m in corpus.metas:
        if m["split"] == "val":
            val_by_task.setdefault(m["task"], []).append(m)

    results = {}
    for task_name, samples in sorted(val_by_task.items()):
        samples = samples[: args.per_task]
        print(f"== {task_name} ({len(samples)} captions) ==", flush=True)
        res = eval_task(task_name, samples, ckpt_path, args.guidance, args.steps, encoder, corpus)
        results[task_name] = res
        print(json.dumps(res["summary"]), flush=True)

    out = Path(args.out) if args.out else ckpt_path.parent / "eval.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)

    # Markdown table
    lines = [
        "| task | generated succ. | teacher succ. | random succ. | gen score | teacher score |",
        "|---|---|---|---|---|---|",
    ]
    for t, r in results.items():
        s = r["summary"]
        lines.append(
            f"| {t} | {s['generated_success']} | {s['teacher_success']} | {s['random_success']} "
            f"| {s['generated_score_mean']} | {s['teacher_score_mean']} |"
        )
    table = "\n".join(lines)
    with open(out.with_suffix(".md"), "w") as f:
        f.write(table + "\n")
    print(table)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
