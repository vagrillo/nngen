"""Diffusion training loss and DDIM sampling with classifier-free guidance."""

from __future__ import annotations

import torch

from ..config import DiffusionConfig
from ..controller import canvas_mask
from .schedule import Schedule

X0_CLAMP = 1.2  # clamp predicted x0 (normalized canvas units) at each DDIM step


class Diffusion:
    def __init__(self, cfg: DiffusionConfig) -> None:
        self.cfg = cfg
        self.schedule = Schedule(cfg.timesteps, cfg.schedule_s)
        self.mask = canvas_mask()  # static validity mask (C, 32, 32)

    def _input(self, xt: torch.Tensor) -> torch.Tensor:
        m = self.mask.to(xt.device).expand(xt.shape[0], -1, -1, -1)
        return torch.cat([xt, m], dim=1)

    def _null_like(self, text: torch.Tensor, text_mask: torch.Tensor, null: tuple[torch.Tensor, torch.Tensor]):
        """Replace caption conditioning with the null prompt (CFG dropout).

        `null` holds the empty-prompt embedding with a batch dim of 1:
        (1, L, D) and (1, L).
        """
        n_emb, n_mask = null
        B = text.shape[0]
        L = text.shape[1]
        if n_emb.shape[1] < L:  # pad null tokens to the batch's caption length
            n_emb = torch.nn.functional.pad(n_emb, (0, 0, 0, L - n_emb.shape[1]))
            n_mask = torch.nn.functional.pad(n_mask, (0, L - n_mask.shape[1]))
        return (
            n_emb.to(text.device).expand(B, -1, -1),
            n_mask.to(text.device).expand(B, -1),
        )

    def training_loss(
        self,
        model,
        canvas: torch.Tensor,
        emb: torch.Tensor,
        emb_mask: torch.Tensor,
        null: tuple[torch.Tensor, torch.Tensor],
        generator: torch.Generator | None = None,
        cfg_dropout: float | None = None,
        t_range: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """v-prediction MSE, masked to valid weight entries.

        `cfg_dropout` overrides DiffusionConfig.cfg_dropout (pass 0.0 for a
        deterministic validation loss). `t_range` restricts the sampled
        timesteps (diagnostics only).
        """
        B = canvas.shape[0]
        if t_range is None:
            t = torch.randint(0, self.cfg.timesteps, (B,), generator=generator)
        else:
            t = torch.randint(t_range[0], t_range[1], (B,), generator=generator)
        noise = torch.randn(canvas.shape, generator=generator)
        xt, v_target = self.schedule.add_noise(canvas, t, noise)

        drop = torch.rand(B, generator=generator) < (self.cfg.cfg_dropout if cfg_dropout is None else cfg_dropout)
        if drop.any():
            n_emb, n_mask = self._null_like(emb, emb_mask, null)
            emb = torch.where(drop[:, None, None], n_emb, emb)
            emb_mask = torch.where(drop[:, None], n_mask, emb_mask)

        v_pred = model(self._input(xt), t, emb, emb_mask)
        valid = self.mask.to(canvas.device).expand_as(canvas)
        return ((v_pred - v_target) ** 2 * valid).sum() / valid.sum()

    @torch.no_grad()
    def sample(
        self,
        model,
        n: int,
        emb: torch.Tensor,
        emb_mask: torch.Tensor,
        null: tuple[torch.Tensor, torch.Tensor],
        guidance: float = 4.0,
        steps: int = 50,
        generator: torch.Generator | None = None,
        device: str = "cpu",
    ) -> torch.Tensor:
        """DDIM sampling of a normalized canvas batch (n, C, 32, 32)."""
        was_training = model.training
        model.eval()
        shape = (n, self.cfg.out_channels, 32, 32)
        xt = torch.randn(shape, generator=generator).to(device)
        path = self.schedule.timestep_path(steps)
        mask = self.mask.to(device).expand(n, -1, -1, -1)

        for t_now, t_next in zip(path[:-1], path[1:]):
            t_batch = torch.full((n,), t_now, dtype=torch.long, device=device)
            v_c = model(self._input(xt), t_batch, emb, emb_mask)
            n_emb, n_mask = self._null_like(emb, emb_mask, null)
            v_u = model(self._input(xt), t_batch, n_emb, n_mask)
            _, eps_c = self.schedule.x0_eps_from_v(xt, v_c, t_batch)
            _, eps_u = self.schedule.x0_eps_from_v(xt, v_u, t_batch)
            eps = eps_u + guidance * (eps_c - eps_u)
            xt = self.schedule.ddim_step(xt, eps, t_batch, t_next)
            if t_next >= 0:
                xt = xt.clamp(-X0_CLAMP, X0_CLAMP)
        if was_training:
            model.train()
        return xt


class EMA:
    """Exponential moving average of model parameters."""

    def __init__(self, model: torch.nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def state_dict_for_model(self) -> dict:
        """EMA weights in model-state_dict form (same dtypes as targets)."""
        return {k: v.clone() for k, v in self.shadow.items()}
