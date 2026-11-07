"""gait: animate a two-joint leg so its foot traces a stride ellipse (IK)."""

from __future__ import annotations

import math

import torch

from .base import BaseTask, DT, TaskSpec, u

L1 = 0.4  # thigh length
L2 = 0.4  # shank length
FOOT_Y = -0.55
ANGLE_MAX = 1.5
DANGLE_MAX = 0.45
WARMUP = 20  # steps ignored in loss/score (transient from the initial pose)


def forward_kinematics(h: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Angles measured from the downward vertical; foot position (B, 2).

    h: hip angle, k: knee angle (relative), each (B, 1). h = k = 0 -> foot down.
    """
    knee = torch.cat([L1 * torch.sin(h), -L1 * torch.cos(h)], dim=-1)
    foot = knee + torch.cat([L2 * torch.sin(h + k), -L2 * torch.cos(h + k)], dim=-1)
    return foot


class GaitTask(BaseTask):
    spec = TaskSpec(
        name="gait",
        obs_used=6,
        act_used=2,
        io_desc="Inputs: gait phase (sin/cos), previous hip and knee angles, stride "
        "ellipse width and height. Outputs: hip and knee angular velocities.",
    )

    def reset(self, batch: int, gen: torch.Generator) -> dict:
        return {
            "h": torch.zeros(batch, 1),
            "k": torch.zeros(batch, 1),
            "omega": u(gen, batch, 1, lo=3.0, hi=5.0),
            "a": u(gen, batch, 1, lo=0.16, hi=0.28),
            "b": u(gen, batch, 1, lo=0.07, hi=0.13),
            "t": torch.zeros(batch, 1),
        }

    def _phase(self, state: dict) -> torch.Tensor:
        return state["omega"] * state["t"]  # (B,1)

    def target(self, state: dict) -> torch.Tensor:
        ph = self._phase(state)
        return torch.cat([state["a"] * torch.sin(ph), FOOT_Y + state["b"] * torch.cos(ph)], dim=-1)

    def dynamics(self, state: dict, action: torch.Tensor) -> dict:
        dh = action[:, :1].clamp(-DANGLE_MAX, DANGLE_MAX)
        dk = action[:, 1:2].clamp(-DANGLE_MAX, DANGLE_MAX)
        h = (state["h"] + dh).clamp(-ANGLE_MAX, ANGLE_MAX)
        k = (state["k"] + dk).clamp(-ANGLE_MAX, ANGLE_MAX)
        return {**state, "h": h, "k": k, "t": state["t"] + DT}

    def observe(self, state: dict) -> torch.Tensor:
        ph = self._phase(state)
        zeros = state["h"] * 0
        return torch.cat(
            [ph.sin(), ph.cos(), state["h"], state["k"], state["a"], state["b"], zeros, zeros], dim=-1
        )

    def _errs(self, states: list[dict]) -> torch.Tensor:
        return torch.stack([(forward_kinematics(st["h"], st["k"]) - self.target(st)).norm(dim=-1) for st in states])

    def train_loss(self, states: list[dict], w_pointwise: float = 0.0) -> torch.Tensor:
        errs = self._errs(states)[WARMUP:]
        # Smoothness: penalize angular jerk (second difference of the angles).
        h = torch.stack([st["h"] for st in states])
        k = torch.stack([st["k"] for st in states])
        jerk = ((h[2:] - 2 * h[1:-1] + h[:-2]) ** 2).mean() + ((k[2:] - 2 * k[1:-1] + k[:-2]) ** 2).mean()
        return errs.mean() + 0.05 * jerk

    def score(self, states: list[dict]) -> torch.Tensor:
        with torch.no_grad():
            return torch.exp(-40.0 * self._errs(states)[WARMUP:].mean(dim=0))

    def success(self, states: list[dict]) -> torch.Tensor:
        with torch.no_grad():
            return self._errs(states)[WARMUP:].mean(dim=0) < 0.05

    def report(self, states: list[dict]) -> dict[str, float]:
        with torch.no_grad():
            return {"foot_err": self._errs(states)[WARMUP:].mean().item()}
