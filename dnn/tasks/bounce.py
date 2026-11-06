"""bounce: move a paddle to intercept a bouncing ball (pong-like gameplay)."""

from __future__ import annotations

import math

import torch

from .base import BaseTask, DT, TaskSpec, u

PI_4 = math.pi / 4

WALL = 0.97
PADDLE_Y = 0.05
PADDLE_HALF = 0.11
PADDLE_XLIM = 0.91
BALL_SPEED = 0.9
V_PADDLE = 2.2
A_PADDLE = 12.0


class BounceTask(BaseTask):
    spec = TaskSpec(
        name="bounce",
        obs_used=6,
        act_used=1,
        io_desc="Inputs: ball position and velocity, paddle position and velocity. "
        "Output: paddle acceleration (horizontal).",
    )

    def reset(self, batch: int, gen: torch.Generator) -> dict:
        angle = u(gen, batch, 1, lo=PI_4 + 0.2, hi=PI_4 + 1.0)  # downward-left/right mix
        sign = torch.where(torch.rand(batch, 1, generator=gen) < 0.5, -1.0, 1.0)
        bvel = sign * torch.cat([torch.sin(angle), -torch.cos(angle)], dim=-1) * BALL_SPEED
        return {
            "bpos": torch.cat([torch.full((batch, 1), 0.5), torch.full((batch, 1), 0.55)], dim=-1),
            "bvel": bvel,
            "px": torch.full((batch, 1), 0.5),
            "pv": torch.zeros(batch, 1),
            "hits": torch.zeros(batch, 1),
            "t": torch.zeros(batch, 1),
        }

    def dynamics(self, state: dict, action: torch.Tensor) -> dict:
        acc = action[:, :1].clamp(-A_PADDLE, A_PADDLE)
        pv = (state["pv"] + acc * DT).clamp(-V_PADDLE, V_PADDLE)
        px = (state["px"] + pv * DT).clamp(PADDLE_HALF, PADDLE_XLIM)

        bvel = state["bvel"].clone()
        bpos = state["bpos"] + bvel * DT
        # Wall reflections (piecewise-linear, gradients flow through branch values).
        bx, by = bpos[:, :1], bpos[:, 1:]
        flip_x = ((bx < 1 - WALL) & (bvel[:, :1] < 0)) | ((bx > WALL) & (bvel[:, :1] > 0))
        bvel[:, :1] = torch.where(flip_x, -bvel[:, :1], bvel[:, :1])
        bpos[:, :1] = bx.clamp(1 - WALL, WALL)

        flip_y = (by > WALL) & (bvel[:, 1:] > 0)
        bvel[:, 1:] = torch.where(flip_y, -bvel[:, 1:], bvel[:, 1:])
        bpos[:, 1:] = torch.where(by > WALL, 2 * WALL - by, bpos[:, 1:])

        # Floor: paddle contact (hit) or plain bounce; always reflect upward.
        at_floor = bpos[:, 1:] < PADDLE_Y
        bpos[:, 1:] = torch.where(at_floor, 2 * PADDLE_Y - bpos[:, 1:], bpos[:, 1:])
        bvel[:, 1:] = torch.where(at_floor & (bvel[:, 1:] < 0), -bvel[:, 1:], bvel[:, 1:])
        contact = at_floor & ((bpos[:, :1] - px).abs() < PADDLE_HALF)
        hits = state["hits"] + contact.float().detach()

        return {**state, "bvel": bvel, "bpos": bpos, "px": px, "pv": pv, "hits": hits, "t": state["t"] + DT}

    def observe(self, state: dict) -> torch.Tensor:
        zeros = state["px"] * 0
        return torch.cat(
            [state["bpos"], state["bvel"], state["px"], state["pv"], zeros, zeros], dim=-1
        )

    def train_loss(self, states: list[dict]) -> torch.Tensor:
        """Dense tracking proxy: keep the paddle under the ball's x whenever it
        descends, with weight growing toward the paddle line."""
        loss = 0.0
        for st in states:
            bx, by = st["bpos"][:, :1], st["bpos"][:, 1:]
            descending = (st["bvel"][:, 1:] < 0).float()
            w = torch.exp(-(((by - 0.12) / 0.4) ** 2)) + 0.2
            err = (bx - st["px"]).abs() * w * descending
            loss = loss + err.mean()
        return loss / len(states)

    def score(self, states: list[dict]) -> torch.Tensor:
        with torch.no_grad():
            return states[-1]["hits"].squeeze(-1)

    def success(self, states: list[dict]) -> torch.Tensor:
        with torch.no_grad():
            return states[-1]["hits"].squeeze(-1) >= 1

    def report(self, states: list[dict]) -> dict[str, float]:
        with torch.no_grad():
            return {"hits": states[-1]["hits"].mean().item()}
