"""seek: steer a point mass to follow a moving target (chase-game behavior)."""

from __future__ import annotations

import math

import torch

from .base import BaseTask, DT, TaskSpec, dist, u


class SeekTask(BaseTask):
    spec = TaskSpec(
        name="seek",
        obs_used=8,
        act_used=2,
        io_desc="Inputs: agent position and velocity, target position and velocity. "
        "Output: 2D acceleration command.",
    )

    VMAX = 1.2
    AMAX = 3.0

    def reset(self, batch: int, gen: torch.Generator) -> dict:
        return {
            "pos": u(gen, batch, 2, lo=-1, hi=1),
            "vel": u(gen, batch, 2, lo=-0.3, hi=0.3),
            "center": u(gen, batch, 2, lo=-0.25, hi=0.25),
            "amp": u(gen, batch, 1, lo=0.35, hi=0.6),
            "freq": u(gen, batch, 2, lo=0.8, hi=1.6),
            "phase": u(gen, batch, 2, lo=0, hi=2 * math.pi),
            "t": torch.zeros(batch, 1),
        }

    def target(self, state: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Target position and velocity at the current time: analytic Lissajous."""
        tt = state["t"]
        arg = state["freq"] * tt + state["phase"]
        pos = state["center"] + state["amp"] * torch.sin(arg)
        vel = state["amp"] * state["freq"] * torch.cos(arg)
        return pos, vel

    def dynamics(self, state: dict, action: torch.Tensor) -> dict:
        acc = action[:, :2].clamp(-self.AMAX, self.AMAX)
        vel = (state["vel"] + acc * DT).clamp(-self.VMAX, self.VMAX)
        pos = (state["pos"] + vel * DT).clamp(-1.0, 1.0)
        return {**state, "vel": vel, "pos": pos, "t": state["t"] + DT}

    def observe(self, state: dict) -> torch.Tensor:
        tpos, tvel = self.target(state)
        return torch.cat([state["pos"], state["vel"], tpos, tvel], dim=-1)

    def _dists(self, states: list[dict]) -> torch.Tensor:
        return torch.stack([dist(st["pos"], self.target(st)[0]) for st in states])  # (T+1, B)

    def train_loss(self, states: list[dict]) -> torch.Tensor:
        return self._dists(states).mean()

    def score(self, states: list[dict]) -> torch.Tensor:
        with torch.no_grad():
            return torch.exp(-2.0 * self._dists(states).mean(dim=0))

    def success(self, states: list[dict]) -> torch.Tensor:
        with torch.no_grad():
            d = self._dists(states)
            return (d.mean(dim=0) < 0.35) & (d[-1] < 0.15)

    def report(self, states: list[dict]) -> dict[str, float]:
        with torch.no_grad():
            d = self._dists(states)
            return {"mean_dist": d.mean().item(), "final_dist": d[-1].mean().item()}
