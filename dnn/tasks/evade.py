"""evade: escape a proportional-pursuit pursuer (survival-game behavior)."""

from __future__ import annotations

import torch

from .base import BaseTask, DT, TaskSpec, dist, u


class EvadeTask(BaseTask):
    spec = TaskSpec(
        name="evade",
        obs_used=8,
        act_used=2,
        io_desc="Inputs: agent position and velocity, pursuer position and velocity. "
        "Output: 2D acceleration command.",
    )

    VMAX = 1.45      # agent is clearly faster than the pursuer
    AMAX = 4.0
    P_SPEED = 1.0    # constant pursuer speed
    CAUGHT = 0.10    # capture distance

    def reset(self, batch: int, gen: torch.Generator) -> dict:
        pos = u(gen, batch, 2, lo=-0.85, hi=0.85)
        pursuer = pos + u(gen, batch, 2, lo=0.5, hi=0.9) * (torch.randint(0, 2, (batch, 2), generator=gen) * 2 - 1)
        pursuer = pursuer.clamp(-0.95, 0.95)
        return {
            "pos": pos,
            "vel": u(gen, batch, 2, lo=-0.2, hi=0.2),
            "pursuer": pursuer,
            "t": torch.zeros(batch, 1),
        }

    def dynamics(self, state: dict, action: torch.Tensor) -> dict:
        acc = action[:, :2].clamp(-self.AMAX, self.AMAX)
        vel = (state["vel"] + acc * DT).clamp(-self.VMAX, self.VMAX)
        pos = (state["pos"] + vel * DT).clamp(-1.0, 1.0)
        # Proportional pursuit at constant speed.
        d = dist(state["pursuer"], pos).clamp_min(1e-6).unsqueeze(-1)
        pv = (pos - state["pursuer"]) / d * self.P_SPEED
        pursuer = state["pursuer"] + pv * DT
        return {**state, "vel": vel, "pos": pos, "pursuer": pursuer, "t": state["t"] + DT}

    def observe(self, state: dict) -> torch.Tensor:
        d = dist(state["pursuer"], state["pos"]).clamp_min(1e-6).unsqueeze(-1)
        pvel = (state["pos"] - state["pursuer"]) / d * self.P_SPEED
        return torch.cat([state["pos"], state["vel"], state["pursuer"], pvel], dim=-1)

    def _dists(self, states: list[dict]) -> torch.Tensor:
        return torch.stack([dist(st["pos"], st["pursuer"]) for st in states])  # (T+1, B)

    def train_loss(self, states: list[dict], w_pointwise: float = 0.0) -> torch.Tensor:
        d = self._dists(states)
        keep_away = torch.relu(0.4 - d).mean()
        survive = torch.relu(self.CAUGHT + 0.03 - d).mean()
        return keep_away + 3.0 * survive

    def score(self, states: list[dict]) -> torch.Tensor:
        with torch.no_grad():
            d = self._dists(states)
            return torch.exp(-6.0 * torch.relu(0.35 - d).mean(dim=0))

    def success(self, states: list[dict]) -> torch.Tensor:
        with torch.no_grad():
            d = self._dists(states)
            return (d.min(dim=0).values > self.CAUGHT) & (d.mean(dim=0) > 0.38)

    def report(self, states: list[dict]) -> dict[str, float]:
        with torch.no_grad():
            d = self._dists(states)
            return {"mean_dist": d.mean().item(), "min_dist": d.min(dim=0).values.mean().item()}
