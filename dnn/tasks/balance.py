"""balance: classic cart-pole stabilization (physics gameplay)."""

from __future__ import annotations

import torch

from .base import BaseTask, DT, TaskSpec, u

GRAV = 9.8
POLE_LEN = 0.5
M_CART = 1.0
M_POLE = 0.1
FMAX = 10.0
UPRIGHT = 0.2095  # ~12 degrees


def _derivs(x, xd, th, thd, f):
    """Standard cart-pole equations; theta measured from upright."""
    total = M_CART + M_POLE
    temp = (f + M_POLE * POLE_LEN * thd * thd * torch.sin(th)) / total
    th_acc = (GRAV * torch.sin(th) + torch.cos(th) * temp) / (
        POLE_LEN * (4.0 / 3.0 - M_POLE * torch.cos(th) * torch.cos(th) / total)
    )
    x_acc = temp - M_POLE * POLE_LEN * th_acc * torch.cos(th) / total
    return x_acc, th_acc


class BalanceTask(BaseTask):
    spec = TaskSpec(
        name="balance",
        obs_used=5,
        act_used=1,
        io_desc="Inputs: cart position and velocity, pole angle (sin/cos) and angular "
        "velocity. Output: horizontal force on the cart.",
    )

    def reset(self, batch: int, gen: torch.Generator) -> dict:
        return {
            "x": u(gen, batch, 1, lo=-0.3, hi=0.3),
            "xd": u(gen, batch, 1, lo=-0.2, hi=0.2),
            "th": u(gen, batch, 1, lo=-0.08, hi=0.08),
            "thd": u(gen, batch, 1, lo=-0.1, hi=0.1),
            "t": torch.zeros(batch, 1),
        }

    def dynamics(self, state: dict, action: torch.Tensor) -> dict:
        f = action[:, :1].clamp(-FMAX, FMAX)
        x_acc, th_acc = _derivs(state["x"], state["xd"], state["th"], state["thd"], f)
        xd = state["xd"] + x_acc * DT
        x = state["x"] + xd * DT
        thd = state["thd"] + th_acc * DT
        th = state["th"] + thd * DT
        x = x.clamp(-1.5, 1.5)
        return {**state, "x": x, "xd": xd, "th": th, "thd": thd, "t": state["t"] + DT}

    def observe(self, state: dict) -> torch.Tensor:
        zeros = state["x"] * 0
        return torch.cat(
            [state["x"], state["xd"], state["th"].sin(), state["th"].cos(), state["thd"], zeros, zeros, zeros],
            dim=-1,
        )

    def train_loss(self, states: list[dict]) -> torch.Tensor:
        loss = 0.0
        for st in states:
            loss = loss + (st["th"] ** 2).mean() + 0.1 * (st["x"] ** 2).mean() + 0.01 * (st["thd"] ** 2).mean()
        return loss / len(states)

    def _upright_frac(self, states: list[dict]) -> torch.Tensor:
        th = torch.stack([st["th"].squeeze(-1) for st in states])       # (T+1, B)
        x = torch.stack([st["x"].squeeze(-1) for st in states])
        ok = (th.abs() < UPRIGHT) & (x.abs() < 1.0)
        return ok.float().mean(dim=0)

    def score(self, states: list[dict]) -> torch.Tensor:
        with torch.no_grad():
            return self._upright_frac(states)

    def success(self, states: list[dict]) -> torch.Tensor:
        with torch.no_grad():
            th = torch.stack([st["th"].squeeze(-1) for st in states[-100:]])
            x = torch.stack([st["x"].squeeze(-1) for st in states[-100:]])
            ok = (th.abs() < UPRIGHT) & (x.abs() < 1.0)
            return ok.float().mean(dim=0) > 0.95

    def report(self, states: list[dict]) -> dict[str, float]:
        with torch.no_grad():
            return {"upright_frac": self._upright_frac(states).mean().item()}
