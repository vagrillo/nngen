"""Visual rendering: behavior GIFs for generated networks and weight heatmaps.

Every task family gets a small matplotlib scene renderer; frames are rendered
off-screen (Agg) and assembled into a GIF with Pillow. Element 0 of the batch
is rendered.
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from .tasks.gait import L1, L2, forward_kinematics  # noqa: E402

PARAM_LABELS = ("W1", "b1", "W2", "b2", "W3", "b3")


def _p(t: torch.Tensor) -> np.ndarray:
    """Detach batch element 0 and squeeze to numpy."""
    return t.detach()[0].cpu().numpy()


def _finish(fig, frames: list[Image.Image]) -> None:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    frames.append(Image.open(buf).convert("RGB").resize((320, 320)))


def _new_fig(xlim: tuple[float, float], ylim: tuple[float, float]):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax


# -- per-family scene painters ------------------------------------------------
def _paint_seek(ax, states, t, task) -> None:
    pos = np.stack([_p(st["pos"]) for st in states[: t + 1]])
    tgt = np.stack([_p(task.target(st)[0]) for st in states[: t + 1]])
    ax.plot(tgt[:, 0], tgt[:, 1], color="#bbbbbb", lw=1, alpha=0.7, label="target path")
    ax.plot(pos[:, 0], pos[:, 1], color="#1f77b4", lw=1.5)
    ax.plot(*_p(states[t]["pos"]), "o", color="#1f77b4", ms=6)
    ax.plot(*_p(task.target(states[t])[0]), "*", color="#d62728", ms=10)
    ax.set_title("seek: agent -> target", fontsize=9)


def _paint_evade(ax, states, t, task) -> None:
    pos = np.stack([_p(st["pos"]) for st in states[: t + 1]])
    purs = np.stack([_p(st["pursuer"]) for st in states[: t + 1]])
    ax.plot(pos[:, 0], pos[:, 1], color="#1f77b4", lw=1.5, label="agent")
    ax.plot(purs[:, 0], purs[:, 1], color="#d62728", lw=1.5, alpha=0.8, label="pursuer")
    ax.plot(*_p(states[t]["pos"]), "o", color="#1f77b4", ms=6)
    ax.plot(*_p(states[t]["pursuer"]), "o", color="#d62728", ms=6)
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title("evade: escape the pursuer", fontsize=9)


def _paint_balance(ax, states, t, task) -> None:
    x = float(_p(states[t]["x"]))
    th = float(_p(states[t]["th"]))
    ax.add_patch(plt.Rectangle((-0.25 + x, -0.08), 0.5, 0.12, color="#444444"))
    wheel_y = -0.08
    ax.plot([x - 0.15, x + 0.15], [wheel_y, wheel_y], "k.", ms=4)
    pole_x = x + 0.5 * np.sin(th)
    pole_y = 0.04 + 0.5 * np.cos(th)
    ax.plot([x, pole_x], [0.04, pole_y], color="#d62728", lw=4, solid_capstyle="round")
    ax.plot([x], [0.04], "o", color="#1f77b4", ms=8)
    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-0.3, 1.1)
    ax.set_title("balance: cart-pole", fontsize=9)


def _paint_bounce(ax, states, t, task) -> None:
    ball = _p(states[t]["bpos"])
    px = float(_p(states[t]["px"]))
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, fill=False, color="#888888"))
    ax.add_patch(plt.Rectangle((px - 0.09, 0), 0.18, 0.05, color="#1f77b4"))
    trail = np.stack([_p(st["bpos"]) for st in states[: t + 1]])
    ax.plot(trail[:, 0], trail[:, 1], color="#ff7f0e", lw=1, alpha=0.5)
    ax.plot(*ball, "o", color="#d62728", ms=8)
    ax.set_title("bounce: paddle intercept", fontsize=9)


def _paint_gait(ax, states, t, task) -> None:
    h = torch.tensor([[_p(states[t]["h"]).item()]])
    k = torch.tensor([[_p(states[t]["k"]).item()]])
    knee = (L1 * torch.sin(h), -L1 * torch.cos(h))
    foot = forward_kinematics(h, k)
    tgt = _p(task.target(states[t]))
    ax.plot(0, 0, "s", color="#444444", ms=8)
    ax.plot([0, float(knee[0][0, 0])], [0, float(knee[1][0, 0])], color="#1f77b4", lw=4)
    ax.plot([float(knee[0][0, 0]), float(foot[0, 0])], [float(knee[1][0, 0]), float(foot[0, 1])], color="#1f77b4", lw=4)
    ax.plot(float(foot[0, 0]), float(foot[0, 1]), "o", color="#1f77b4", ms=6)
    ph = np.linspace(0, 2 * np.pi, 100)
    a = float(_p(states[t]["a"]))
    b = float(_p(states[t]["b"]))
    ax.plot(a * np.sin(ph), -0.55 + b * np.cos(ph), color="#d62728", lw=1, ls="--", label="stride target")
    ax.legend(loc="upper right", fontsize=6)
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.95, 0.15)
    ax.set_title("gait: leg IK animation", fontsize=9)


def _paint_orbit(ax, states, t, task) -> None:
    pos = np.stack([_p(st["pos"]) for st in states[: t + 1]])
    r0 = float(_p(states[0]["r0"]))
    ax.add_patch(plt.Circle((0, 0), 0.08, color="#444444"))
    ax.add_patch(plt.Circle((0, 0), r0, fill=False, color="#d62728", ls="--", lw=1))
    ax.plot(pos[:, 0], pos[:, 1], color="#1f77b4", lw=1.2)
    ax.plot(*_p(states[t]["pos"]), "o", color="#1f77b4", ms=6)
    ax.set_title(f"orbit: r0={r0:.2f}", fontsize=9)


def _paint_draw(ax, states, t, task) -> None:
    pen = np.stack([_p(st["pen"]) for st in states[: t + 1]])
    target = _p(states[0]["target"])
    ax.plot(target[:, 0], target[:, 1], color="#d62728", lw=1.5, ls=":", label="target")
    ax.plot(pen[:, 0], pen[:, 1], color="#1f77b4", lw=1.8, label="pen")
    ax.plot(*_p(states[t]["pen"]), "o", color="#1f77b4", ms=5)
    ax.legend(loc="upper right", fontsize=6)
    ax.set_title(f"draw: {task.shape}", fontsize=9)


_PAINTERS = {
    "seek": _paint_seek,
    "evade": _paint_evade,
    "balance": _paint_balance,
    "bounce": _paint_bounce,
    "gait": _paint_gait,
    "orbit": _paint_orbit,
}


def render_gif(task_name: str, task, states: list[dict], out_path: str | Path, fps: int = 24) -> Path:
    """Render a rollout to an animated GIF (batch element 0)."""
    frames: list[Image.Image] = []
    steps = len(states)
    every = max(1, steps // 48)
    if task_name.startswith("draw"):
        painter = _paint_draw
    else:
        painter = _PAINTERS[task_name]
    for t in range(0, steps, every):
        fig, ax = _new_fig((-1.25, 1.25), (-1.25, 1.25))
        if task_name == "balance":
            fig, ax = _new_fig((-1.6, 1.6), (-0.3, 1.1))
        if task_name == "bounce":
            fig, ax = _new_fig((-0.05, 1.05), (-0.05, 1.05))
        if task_name == "gait":
            fig, ax = _new_fig((-0.5, 0.5), (-0.95, 0.15))
        painter(ax, states, t, task)
        _finish(fig, frames)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0)
    return out


def render_weights(canvas: torch.Tensor, out_path: str | Path) -> Path:
    """Heatmap of the 6 canvas channels (raw weight space) + validity mask."""
    c = canvas.detach().cpu().numpy()
    fig, axes = plt.subplots(1, 6, figsize=(12, 2.4))
    for i, ax in enumerate(axes):
        im = ax.imshow(c[i], cmap="viridis")
        ax.set_title(PARAM_LABELS[i], fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Generated weight canvas (padded regions are zero)", fontsize=10)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", dpi=110)
    plt.close(fig)
    return out
