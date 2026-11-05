"""Cosine noise schedule and v-prediction algebra.

v-prediction: the model predicts v = a*eps - s*x0 where a = sqrt(alphas_cumprod)
and s = sqrt(1 - alphas_cumprod). v is bounded for both x0 and eps extremes,
which trains more stably than raw epsilon prediction.
"""

from __future__ import annotations

import math

import torch


class Schedule:
    def __init__(self, timesteps: int, s: float = 0.008) -> None:
        self.T = timesteps
        t = torch.linspace(0, 1, timesteps + 1)
        alpha_bar = ((t + s) / (1 + s) * math.pi / 2).cos() ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        # betas from alpha_bar ratios; keep them in a sane range
        betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(1e-5, 0.999)
        acp = torch.cumprod(1 - betas, dim=0)                    # alphas_cumprod (T,)
        self.betas = betas
        self.alphas_cumprod = acp
        self.a = acp.sqrt()                                      # signal sqrt
        self.s = (1 - acp).sqrt()                                # noise sqrt

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        """xt, v_target for given timesteps t (B,)."""
        a = self.a[t].view(-1, 1, 1, 1)
        s = self.s[t].view(-1, 1, 1, 1)
        xt = a * x0 + s * noise
        v = a * noise - s * x0
        return xt, v

    def x0_eps_from_v(self, xt: torch.Tensor, v: torch.Tensor, t: torch.Tensor):
        a = self.a[t].view(-1, 1, 1, 1)
        s = self.s[t].view(-1, 1, 1, 1)
        x0 = a * xt - s * v
        eps = s * xt + a * v
        return x0, eps

    def ddim_step(self, xt: torch.Tensor, eps: torch.Tensor, t_now: torch.Tensor, t_next: int) -> torch.Tensor:
        """Deterministic DDIM update (eta=0). `t_now` is a (B,) tensor;
        `t_next` a python int, -1 meaning the final step (return x0)."""
        a_now = self.a[t_now].view(-1, 1, 1, 1)
        s_now = self.s[t_now].view(-1, 1, 1, 1)
        x0 = (xt - s_now * eps) / a_now
        if t_next < 0:  # jump straight to x0
            return x0
        a_next = self.a[t_next].view(1, 1, 1, 1)
        s_next = self.s[t_next].view(1, 1, 1, 1)
        return a_next * x0 + s_next * eps

    def timestep_path(self, steps: int) -> list[int]:
        """Descending timestep path for DDIM sampling: [t0, ..., tk, -1].

        Iterate with `for t_now, t_next in zip(path[:-1], path[1:])`.
        """
        ts = torch.linspace(0, self.T - 1, steps).round().long()
        return ts.flip(0).tolist() + [-1]
