"""Tests for the canonical controller and weight canvas encoding."""

import torch

from dnn.controller import (
    BatchedController,
    Controller,
    NUM_PARAMS,
    ParamStats,
    canvas_mask,
    canvas_to_flat,
    controller_from_flat,
    denormalize_canvas,
    flat_to_canvas,
    mask_flat,
    normalize_canvas,
)
from dnn.config import CANVAS, NUM_CHANNELS


def test_mask_covers_exactly_num_params():
    mask = canvas_mask()
    assert mask.shape == (NUM_CHANNELS, CANVAS, CANVAS)
    assert mask.sum().item() == NUM_PARAMS


def test_canvas_round_trip():
    flat = torch.randn(7, NUM_PARAMS)
    canvas = flat_to_canvas(flat)
    assert canvas.shape == (7, NUM_CHANNELS, CANVAS, CANVAS)
    # All entries outside the mask are exactly zero.
    assert canvas[canvas_mask().expand_as(canvas).bool() == False].abs().sum() == 0
    back = canvas_to_flat(canvas)
    assert torch.allclose(back, flat, atol=1e-6)


def test_normalize_denormalize_round_trip():
    # Uniform values keep every z-score well inside the +-NORM_RANGE clamp.
    flat = torch.rand(64, NUM_PARAMS) * 0.2 - 0.1
    stats = ParamStats.from_flat(flat)
    canvas = flat_to_canvas(flat)
    norm = normalize_canvas(canvas, stats)
    # Normalized valid entries live in [-1, 1]; padding is zero.
    m = canvas_mask()
    assert norm[:, m.bool()].abs().max() <= 1.0 + 1e-6
    assert norm[:, m.bool() == False].abs().sum() == 0
    raw = denormalize_canvas(norm, stats)
    # Round trip recovers raw values at valid entries.
    assert torch.allclose(raw[m.expand_as(raw).bool()], canvas[m.expand_as(canvas).bool()], atol=1e-5)


def test_controller_forward_and_set_get():
    net = Controller()
    x = torch.randn(5, 8)
    y = net(x)
    assert y.shape == (5, 8)
    flat = net.get_flat()
    assert flat.shape == (NUM_PARAMS,)
    net2 = controller_from_flat(flat)
    assert torch.allclose(net2(x), y, atol=1e-6)


def test_batched_matches_single():
    torch.manual_seed(1)
    B = 6
    bnet = BatchedController(B)
    x = torch.randn(B, 8)
    y = bnet(x)
    assert y.shape == (B, 8)
    flat = bnet.get_flat()
    for i in range(B):
        single = controller_from_flat(flat[i])
        assert torch.allclose(single(x[i : i + 1]), y[i : i + 1], atol=1e-6)
