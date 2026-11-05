"""Tests for schedule, UNet and the diffusion sampler."""

import torch

from dnn.config import get_preset
from dnn.diffusion.schedule import Schedule
from dnn.diffusion.sampler import Diffusion
from dnn.diffusion.unet import UNet


def _fake_batch(B, null, gen):
    canvas = torch.rand(B, 6, 32, 32, generator=gen) * 2 - 1
    emb = torch.randn(B, 12, 384, generator=gen)
    mask = torch.ones(B, 12)
    return canvas, emb, mask, null


def test_cosine_schedule_properties():
    sch = Schedule(1000)
    assert sch.alphas_cumprod[0] > 0.99
    assert sch.alphas_cumprod[-1] < 0.05
    assert (sch.alphas_cumprod[1:] <= sch.alphas_cumprod[:-1]).all()


def test_v_prediction_round_trip():
    sch = Schedule(1000)
    x0 = torch.randn(3, 6, 32, 32)
    t = torch.tensor([0, 500, 999])
    noise = torch.randn_like(x0)
    xt, v = sch.add_noise(x0, t, noise)
    x0_hat, eps_hat = sch.x0_eps_from_v(xt, v, t)
    assert torch.allclose(x0_hat, x0, atol=1e-4)
    assert torch.allclose(eps_hat, noise, atol=1e-4)


def test_unet_forward_shapes_and_grads():
    cfg = get_preset("smoke").model
    model = UNet(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    assert 1e6 < n_params < 8e6, n_params
    x = torch.randn(2, cfg.in_channels, 32, 32)
    t = torch.tensor([10, 900])
    text = torch.randn(2, 16, cfg.text_dim)
    tmask = torch.ones(2, 16)
    out = model(x, t, text, tmask)
    assert out.shape == (2, cfg.out_channels, 32, 32)
    out.mean().backward()
    assert all(p.grad is not None for p in model.parameters())


def test_training_loss_backward_and_cfg_dropout():
    preset = get_preset("smoke")
    model = UNet(preset.model)
    diff = Diffusion(preset.model)
    gen = torch.Generator().manual_seed(0)
    null = (torch.randn(1, 8, 384), torch.ones(1, 8))
    canvas, emb, mask, null = _fake_batch(4, null, gen)
    loss = diff.training_loss(model, canvas, emb, mask, null, generator=gen)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_sample_shapes_and_determinism():
    preset = get_preset("smoke")
    model = UNet(preset.model).eval()
    diff = Diffusion(preset.model)
    null = (torch.randn(1, 8, 384), torch.ones(1, 8))
    emb = torch.randn(2, 8, 384)
    mask = torch.ones(2, 8)
    g1 = torch.Generator().manual_seed(5)
    g2 = torch.Generator().manual_seed(5)
    s1 = diff.sample(model, 2, emb, mask, null, guidance=3.0, steps=4, generator=g1)
    s2 = diff.sample(model, 2, emb, mask, null, guidance=3.0, steps=4, generator=g2)
    assert s1.shape == (2, 6, 32, 32)
    assert torch.allclose(s1, s2)


def test_overfit_tiny_dataset():
    """The model must be able to memorize a handful of canvases.

    Full-range t makes single-step loss variance swamp the trend, so the
    test overfits within a narrow timestep band and compares window means.
    """
    preset = get_preset("smoke")
    model = UNet(preset.model)
    diff = Diffusion(preset.model)
    gen = torch.Generator().manual_seed(1)
    null = (torch.randn(1, 8, 384), torch.ones(1, 8))
    canvas, emb, mask, null = _fake_batch(8, null, gen)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)

    def window_mean(n: int) -> float:
        vals = []
        for _ in range(n):
            loss = diff.training_loss(
                model, canvas, emb, mask, null, generator=gen, cfg_dropout=0.0, t_range=(200, 260)
            )
            vals.append(loss.item())
        return sum(vals) / len(vals)

    early = window_mean(10)
    for step in range(150):
        loss = diff.training_loss(model, canvas, emb, mask, null, generator=gen, t_range=(200, 260))
        opt.zero_grad()
        loss.backward()
        opt.step()
    late = window_mean(10)
    assert late < early * 0.4, (early, late)
