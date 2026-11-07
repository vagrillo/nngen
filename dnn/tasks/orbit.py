"""orbit: thrust a satellite into a circular orbit of a requested radius."""

from __future__ import annotations

import torch

from .base import BaseTask, DT, TaskSpec, u

MU = 1.0
THRUST_MAX = 0.6
CRASH_R = 0.18


class OrbitTask(BaseTask):
    spec = TaskSpec(
        name="orbit",
        obs_used=6,
        act_used=2,
        io_desc="Inputs: satellite position and velocity relative to the planet, "
        "target radius and its inverse. Output: 2D thrust acceleration.",
    )

    def reset(self, batch: int, gen: torch.Generator) -> dict:
        r0 = u(gen, batch, 1, lo=0.45, hi=0.75)
        v_circ = (MU / r0).sqrt()
        # Start near-circular with a deliberate radial/velocity error to correct.
        pos = torch.cat([r0, torch.zeros_like(r0)], dim=-1)
        vel = torch.cat(
            [u(gen, batch, 1, lo=-0.08, hi=0.08), v_circ * u(gen, batch, 1, lo=0.75, hi=1.1)], dim=-1
        )
        return {"pos": pos, "vel": vel, "r0": r0, "t": torch.zeros(batch, 1)}

    def _radius(self, state: dict) -> torch.Tensor:
        return state["pos"].norm(dim=-1, keepdim=True)

    def _v_tangential(self, state: dict) -> torch.Tensor:
        r = self._radius(state).clamp_min(1e-6)
        tangent = torch.cat([-state["pos"][:, 1:], state["pos"][:, :1]], dim=-1) / r
        return (state["vel"] * tangent).sum(dim=-1, keepdim=True)

    def dynamics(self, state: dict, action: torch.Tensor) -> dict:
        thrust = action[:, :2].clamp(-THRUST_MAX, THRUST_MAX)
        r = self._radius(state).clamp_min(0.05)
        gravity = -MU * state["pos"] / (r ** 3)
        acc = gravity + thrust
        vel = state["vel"] + acc * DT
        pos = state["pos"] + vel * DT
        return {**state, "vel": vel, "pos": pos, "t": state["t"] + DT}

    def observe(self, state: dict) -> torch.Tensor:
        zeros = state["r0"] * 0
        return torch.cat([state["pos"], state["vel"], state["r0"], 1.0 / state["r0"], zeros, zeros], dim=-1)

    def _errors(self, states: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        r = torch.stack([self._radius(st).squeeze(-1) for st in states])  # (T+1, B)
        vt = torch.stack([self._v_tangential(st).squeeze(-1) for st in states])
        r0 = torch.stack([st["r0"].squeeze(-1) for st in states])
        v_c = (MU / r0).sqrt()
        return (r - r0).abs(), (vt - v_c).abs()

    def train_loss(self, states: list[dict], w_pointwise: float = 0.0) -> torch.Tensor:
        dr, dvt = self._errors(states)
        crash = torch.relu(CRASH_R - torch.stack([self._radius(st).squeeze(-1) for st in states])).mean()
        return (dr ** 2).mean() + 0.3 * (dvt ** 2).mean() + 10.0 * crash

    def score(self, states: list[dict]) -> torch.Tensor:
        with torch.no_grad():
            dr, dvt = self._errors(states)
            return torch.exp(-3.0 * (dr.mean(dim=0) + 0.5 * dvt.mean(dim=0)))

    def success(self, states: list[dict]) -> torch.Tensor:
        with torch.no_grad():
            dr, dvt = self._errors(states)
            r_min = torch.stack([self._radius(st).squeeze(-1) for st in states]).min(dim=0).values
            return (dr.mean(dim=0) < 0.06) & (dvt.mean(dim=0) < 0.12) & (r_min > CRASH_R)

    def report(self, states: list[dict]) -> dict[str, float]:
        with torch.no_grad():
            dr, dvt = self._errors(states)
            return {"radius_err": dr.mean().item(), "tangential_err": dvt.mean().item()}
