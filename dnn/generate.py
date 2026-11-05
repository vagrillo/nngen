"""Generate a brand-new neural network from a text prompt and run it.

This is the end product: a model that generates models.

    python -m dnn.generate \
        --prompt "A tiny MLP controller that drives a pen to trace a five-pointed star" \
        --task draw_star --ckpt runs/small/ckpt.pt --out runs/demo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from .config import DiffusionConfig
from .controller import Controller, ParamStats, controller_from_flat, denormalize_canvas, flat_to_canvas
from .diffusion.sampler import Diffusion
from .diffusion.unet import UNet
from .render import render_gif, render_weights
from .tasks import get_task, task_names
from .text_encoder import TextEncoder
from .train_diffusion import load_checkpoint


def generate_canvas(
    prompt: str,
    ckpt_path: str | Path,
    guidance: float,
    steps: int,
    seed: int,
    encoder: TextEncoder | None = None,
) -> tuple[torch.Tensor, ParamStats, dict]:
    """Sample a normalized canvas for the prompt; returns (canvas, stats, info)."""
    ckpt = load_checkpoint(Path(ckpt_path))
    model_cfg = ckpt["model_cfg"]
    stats = ParamStats.from_json(ckpt["stats"])
    model = UNet(DiffusionConfig(**model_cfg))
    model.load_state_dict({k: v.to(torch.float32) for k, v in ckpt["ema"].items()})
    model.eval()

    encoder = encoder or TextEncoder()
    emb, mask = encoder.encode([prompt])
    null = encoder.encode([""])  # CFG null prompt

    diff = Diffusion(DiffusionConfig(**model_cfg))
    gen = torch.Generator().manual_seed(seed)
    canvas = diff.sample(model, 1, emb, mask, null, guidance=guidance, steps=steps, generator=gen)
    info = {"prompt": prompt, "ckpt": str(ckpt_path), "guidance": guidance, "steps": steps, "seed": seed}
    return canvas, stats, info


def canvas_to_controller(canvas: torch.Tensor, stats: ParamStats) -> Controller:
    raw = denormalize_canvas(canvas, stats)
    flat = raw.flatten(1)[0]
    return controller_from_flat(flat)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--task", default=None, help=f"task for rollout/render: {', '.join(task_names())}")
    ap.add_argument("--ckpt", default="runs/small/ckpt.pt")
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/demo")
    ap.add_argument("--no-gif", action="store_true")
    args = ap.parse_args()

    task_name = args.task
    if task_name is None:
        raise SystemExit("--task is required (which environment should run the generated network?)")
    task = get_task(task_name)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    canvas, stats, info = generate_canvas(args.prompt, args.ckpt, args.guidance, args.steps, args.seed)
    net = canvas_to_controller(canvas, stats)

    # Evaluate on a held-out episode seed.
    eval_seed = 777_000 + args.seed
    states = task.rollout(net, 1, eval_seed)
    score = task.score(states)
    try:
        success = task.success(states)
        success = bool(success[0])
    except NotImplementedError:
        success = None
    metrics = {k: round(v, 4) for k, v in task.report(states).items()}

    save_file({"flat": net.get_flat()}, out / "generated_net.safetensors")
    render_weights(flat_to_canvas(net.get_flat().unsqueeze(0))[0], out / "weights.png")
    if not args.no_gif:
        gif = render_gif(task_name, task, states, out / "behavior.gif")
        info["gif"] = str(gif)

    report = {
        **info,
        "task": task_name,
        "score": round(float(score[0]), 4),
        "success": success,
        "metrics": metrics,
        "eval_seed": eval_seed,
        "outputs": {"net": str(out / "generated_net.safetensors"), "weights_png": str(out / "weights.png")},
    }
    with open(out / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
