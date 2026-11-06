"""draw: a turtle-graphics pen agent that traces a target shape on a canvas.

One task entry per shape (draw_circle, draw_square, draw_triangle, draw_star,
draw_spiral). The controller must "know" the shape purely from its weights:
observations carry only pen pose, progress, step length and the target size.

Target polylines are unit-size curves resampled to one point per env step and
scaled per batch element, so the pen starts exactly on the target's first
point with the correct initial heading.
"""

from __future__ import annotations

import math

import torch

from .base import BaseTask, TaskSpec, u

STEPS = 120
SIZE_LO, SIZE_HI = 0.5, 0.75
TURN_MAX = 1.2


def _resample_closed(pts: torch.Tensor, n: int) -> torch.Tensor:
    """Resample a closed polyline (vertices) to n uniform-arclength points."""
    loop = torch.cat([pts, pts[:1]], dim=0)
    return _resample(loop, n)


def _resample(loop: torch.Tensor, n: int) -> torch.Tensor:
    seg = loop[1:] - loop[:-1]
    lens = seg.norm(dim=-1)
    cum = torch.cat([torch.zeros(1), lens.cumsum(0)])
    total = cum[-1]
    targets = torch.linspace(0, total, n + 1)[:-1]
    idx = torch.searchsorted(cum, targets, right=True) - 1
    idx = idx.clamp(0, len(seg) - 1)
    seg_len = (cum[idx + 1] - cum[idx]).clamp_min(1e-9)
    frac = ((targets - cum[idx]) / seg_len).unsqueeze(-1)
    return loop[idx] + frac * seg[idx]


def _regular_polygon(circumradius: float, sides: int, rotation: float) -> torch.Tensor:
    ang = rotation + torch.linspace(0, 2 * math.pi, sides + 1)[:-1]
    return circumradius * torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)


def _star(outer: float, inner: float, points: int) -> torch.Tensor:
    verts = []
    for i in range(2 * points):
        r = outer if i % 2 == 0 else inner
        ang = math.pi / 2 + i * math.pi / points
        verts.append([r * math.cos(ang), r * math.sin(ang)])
    return torch.tensor(verts)


def _spiral(n: int) -> torch.Tensor:
    u_ = torch.linspace(0.0, 1.0, n)
    r = u_
    th = 2.5 * 2 * math.pi * u_
    return torch.stack([r * torch.cos(th), r * torch.sin(th)], dim=-1)


def _unit_polyline(shape: str, n: int = STEPS) -> torch.Tensor:
    """Unit-size target polyline, n points, uniform in arclength."""
    if shape == "circle":
        th = torch.linspace(0, 2 * math.pi, n + 1)[:-1]
        pts = torch.stack([torch.cos(th), torch.sin(th)], dim=-1)
    elif shape == "square":
        pts = _resample_closed(_regular_polygon(2 ** 0.5 / 2, 4, math.pi / 4), n)
    elif shape == "triangle":
        pts = _resample_closed(_regular_polygon(0.6, 3, math.pi / 2), n)
    elif shape == "pentagon":
        pts = _resample_closed(_regular_polygon(0.6, 5, math.pi / 2), n)
    elif shape == "hexagon":
        pts = _resample_closed(_regular_polygon(0.6, 6, 0.0), n)
    elif shape == "octagon":
        pts = _resample_closed(_regular_polygon(0.6, 8, math.pi / 8), n)
    elif shape == "star":
        pts = _resample_closed(_star(0.6, 0.24, 5), n)
    elif shape == "spiral":
        # Resample by arclength so steps are uniform like the other shapes.
        pts = _resample(_spiral(400), n)
    elif shape == "line":
        pts = torch.stack([torch.linspace(-0.5, 0.5, n), torch.zeros(n)], dim=-1)
    elif shape == "arc":
        th = torch.linspace(math.pi, 2 * math.pi, n)  # upper half circle
        pts = 0.6 * torch.stack([torch.cos(th), torch.sin(th)], dim=-1)
    else:
        raise ValueError(f"Unknown draw shape '{shape}'")
    return pts


CLOSED_SHAPES = {"circle", "square", "triangle", "pentagon", "hexagon", "octagon", "star"}
SHAPE_NAMES = ("circle", "square", "triangle", "pentagon", "star", "spiral", "line", "arc")
# Held-out shapes: registered for eval only, NEVER generated into the corpus.
HELD_OUT_SHAPES = ("hexagon", "octagon")
_UNIT_CACHE: dict[str, torch.Tensor] = {}


def unit_polyline(shape: str) -> torch.Tensor:
    if shape not in _UNIT_CACHE:
        _UNIT_CACHE[shape] = _unit_polyline(shape)
    return _UNIT_CACHE[shape]


class DrawTask(BaseTask):
    """Pen-agent environment for one shape. Instances live in the registry."""

    def __init__(self, shape: str) -> None:
        self.shape = shape
        self.unit = unit_polyline(shape)          # (STEPS, 2)
        self.closed = shape in CLOSED_SHAPES
        self.start_heading = torch.atan2(
            (self.unit[1] - self.unit[0])[1], (self.unit[1] - self.unit[0])[0]
        )
        self.unit_len = (self.unit[1:] - self.unit[:-1]).norm(dim=-1).sum().item()
        self.spec = TaskSpec(
            name=f"draw_{shape}",
            obs_used=7,
            act_used=2,
            steps=STEPS,
            io_desc="Inputs: pen position, heading (sin/cos), drawing progress, base "
            "step length and target size. Outputs: turn angle and step-length scale.",
        )

    def reset(self, batch: int, gen: torch.Generator) -> dict:
        size = u(gen, batch, 1, lo=SIZE_LO, hi=SIZE_HI)
        target = self.unit.unsqueeze(0) * size.unsqueeze(-1)      # (B, STEPS, 2)
        pen = target[:, 0].clone()
        heading = self.start_heading.expand(batch, 1).clone()
        ds_base = self.unit_len * size / STEPS                    # (B,1)
        return {
            "pen": pen,
            "heading": heading,
            "size": size,
            "ds_base": ds_base,
            "target": target,
            "i": torch.zeros(batch, 1),
            "t": torch.zeros(batch, 1),
        }

    def dynamics(self, state: dict, action: torch.Tensor) -> dict:
        turn = action[:, :1].clamp(-TURN_MAX, TURN_MAX)
        scale = 0.6 + 0.8 * torch.sigmoid(action[:, 1:2])         # step-length factor in [0.6, 1.4]
        heading = state["heading"] + turn
        ds = state["ds_base"] * scale
        pen = state["pen"] + ds * torch.cat([heading.cos(), heading.sin()], dim=-1)
        return {**state, "pen": pen, "heading": heading, "i": state["i"] + 1, "t": state["t"] + 1 / 30}

    def observe(self, state: dict) -> torch.Tensor:
        i = state["i"]
        progress = i / STEPS
        zeros = state["size"] * 0
        return torch.cat(
            [
                state["pen"],
                state["heading"].sin(),
                state["heading"].cos(),
                progress,
                state["ds_base"] * 20.0,
                state["size"],
                zeros,
            ],
            dim=-1,
        )

    def _errors(self, states: list[dict]) -> torch.Tensor:
        """||pen_i - target_i|| for every state (states[0] is exact). (T+1, B)"""
        errs = []
        for st in states:
            idx = st["i"].long().clamp(max=STEPS - 1)
            tgt = st["target"].gather(1, idx.unsqueeze(-1).expand(-1, 1, 2)).squeeze(1)
            errs.append((st["pen"] - tgt).norm(dim=-1))
        return torch.stack(errs)

    def _closure(self, states: list[dict]) -> torch.Tensor:
        return (states[-1]["pen"] - states[-1]["target"][:, 0]).norm(dim=-1)

    def train_loss(self, states: list[dict]) -> torch.Tensor:
        errs = self._errors(states)[1:]
        loss = errs.mean()
        if self.closed:
            loss = loss + 0.5 * self._closure(states).mean()
        return loss

    def score(self, states: list[dict]) -> torch.Tensor:
        with torch.no_grad():
            err = self._errors(states)[1:].mean(dim=0)
            penalty = 0.3 * self._closure(states) if self.closed else 0.0
            return torch.exp(-12.0 * (err + penalty))

    def success(self, states: list[dict]) -> torch.Tensor:
        with torch.no_grad():
            err = self._errors(states)[1:].mean(dim=0)
            ok = err < 0.07
            if self.closed:
                ok = ok & (self._closure(states) < 0.15)
            return ok

    def report(self, states: list[dict]) -> dict[str, float]:
        with torch.no_grad():
            out = {"trace_err": self._errors(states)[1:].mean().item()}
            if self.closed:
                out["closure_err"] = self._closure(states).mean().item()
            return out
