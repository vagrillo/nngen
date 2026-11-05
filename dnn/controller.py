"""The canonical controller network and its weight "canvas".

The diffusion model denoises a fixed-size tensor ("image") whose content is
the parameter set of the canonical controller MLP:

    8 -> 32 (tanh) -> 32 (tanh) -> 8 (linear)     ~1.6k parameters

The six parameter tensors are laid out as channels of a 32x32 canvas: weight
matrices occupy the top-left block, bias vectors occupy the first column.
A static boolean mask marks the valid entries. This mirrors a text2image
pipeline with a fixed resolution: same shape in, same shape out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import CANVAS, HID_DIM, IN_DIM, NUM_CHANNELS, OUT_DIM

# Canonical parameter tensors, in order: (out_features, in_features) like
# nn.Linear weights, or (n,) for biases.
PARAM_SHAPES: tuple[tuple[int, ...], ...] = (
    (HID_DIM, IN_DIM),   # W1
    (HID_DIM,),          # b1
    (HID_DIM, HID_DIM),  # W2
    (HID_DIM,),          # b2
    (OUT_DIM, HID_DIM),  # W3
    (OUT_DIM,),          # b3
)
PARAM_NAMES: tuple[str, ...] = ("W1", "b1", "W2", "b2", "W3", "b3")
NUM_PARAMS = sum(int(torch.Size(s).numel()) for s in PARAM_SHAPES)
assert NUM_PARAMS == 1608

# Fan-in used for the U(-1/sqrt(in), 1/sqrt(in)) init of each tensor
# (weights use their own fan-in; a bias uses the fan-in of its layer).
_TENSOR_FAN_IN: tuple[int, ...] = (IN_DIM, IN_DIM, HID_DIM, HID_DIM, HID_DIM, HID_DIM)


def placed_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    """(rows, cols) of the canvas region occupied by a canonical tensor.

    Weight matrices (out, in) occupy the top-left block; bias vectors (n,)
    occupy the first column, rows 0..n-1.
    """
    if len(shape) == 1:
        return int(shape[0]), 1
    return int(shape[0]), int(shape[1])


def _uniform(n: int, fan_in: int, rng: torch.Generator) -> torch.Tensor:
    bound = 1.0 / math.sqrt(fan_in)
    return torch.empty(n).uniform_(-bound, bound, generator=rng)


def init_params(rng: torch.Generator) -> list[torch.Tensor]:
    """Freshly initialized canonical parameter list (nn.Linear-style bounds)."""
    return [
        _uniform(int(torch.Size(s).numel()), f, rng).reshape(s)
        for s, f in zip(PARAM_SHAPES, _TENSOR_FAN_IN)
    ]


class Controller(nn.Module):
    """Single canonical MLP controller, batch-first: (B, IN_DIM) -> (B, OUT_DIM)."""

    def __init__(self) -> None:
        super().__init__()
        init = init_params(torch.Generator().manual_seed(0))
        self.params = nn.ParameterList([nn.Parameter(t) for t in init])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W1, b1, W2, b2, W3, b3 = self.params
        h = torch.tanh(torch.nn.functional.linear(x, W1, b1))
        h = torch.tanh(torch.nn.functional.linear(h, W2, b2))
        return torch.nn.functional.linear(h, W3, b3)

    # -- parameter plumbing -------------------------------------------------
    def get_flat(self) -> torch.Tensor:
        """All parameters as a flat vector in canonical order, shape (NUM_PARAMS,)."""
        return torch.cat([p.detach().reshape(-1) for p in self.params])

    def set_flat(self, flat: torch.Tensor) -> None:
        """Load parameters from a flat vector in canonical order."""
        with torch.no_grad():
            off = 0
            for p, shape in zip(self.params, PARAM_SHAPES):
                n = int(torch.Size(shape).numel())
                p.copy_(flat[off : off + n].reshape(shape))
                off += n


class BatchedController(nn.Module):
    """A batch of independent canonical controllers.

    Parameter tensors carry a leading batch dimension, so `B` networks are
    optimized in parallel with a single autograd graph (used to mass-produce
    the training corpus). forward takes (B, IN_DIM) -> (B, OUT_DIM).
    """

    def __init__(self, batch: int, seed: int | None = None) -> None:
        super().__init__()
        self.batch = batch
        init = self._fresh_init(0 if seed is None else seed)
        self.params = nn.ParameterList([nn.Parameter(p) for p in init])

    def _fresh_init(self, seed: int) -> list[torch.Tensor]:
        rng = torch.Generator().manual_seed(seed)
        init = init_params(rng)
        grown = []
        for t, shape in zip(init, PARAM_SHAPES):
            s = torch.empty(self.batch, 1, 1) if len(shape) == 2 else torch.empty(self.batch, 1)
            s.uniform_(0.6, 1.5, generator=rng)
            grown.append((t.unsqueeze(0).expand(self.batch, *shape) * s).contiguous())
        return grown

    def reinit_(self, seed: int) -> None:
        """Overwrite parameters in place with a fresh init (keeps graph/compile caches)."""
        with torch.no_grad():
            for p, new in zip(self.params, self._fresh_init(seed)):
                p.copy_(new)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W1, b1, W2, b2, W3, b3 = self.params
        h = torch.tanh(torch.einsum("boi,bi->bo", W1, x) + b1)
        h = torch.tanh(torch.einsum("boi,bi->bo", W2, h) + b2)
        return torch.einsum("boi,bi->bo", W3, h) + b3

    def get_flat(self) -> torch.Tensor:
        """(B, NUM_PARAMS) flat parameters in canonical order."""
        return torch.cat([p.reshape(p.shape[0], -1) for p in self.params], dim=1)


# ---------------------------------------------------------------------------
# Canvas encoding: flat weights <-> (NUM_CHANNELS, CANVAS, CANVAS) image
# ---------------------------------------------------------------------------
_MASK_CACHE: torch.Tensor | None = None


def canvas_mask() -> torch.Tensor:
    """Static validity mask, shape (NUM_CHANNELS, CANVAS, CANVAS), float {0,1}."""
    global _MASK_CACHE
    if _MASK_CACHE is None:
        m = torch.zeros(NUM_CHANNELS, CANVAS, CANVAS)
        for c, shape in enumerate(PARAM_SHAPES):
            rows, cols = placed_shape(shape)
            m[c, :rows, :cols] = 1.0
        _MASK_CACHE = m
    return _MASK_CACHE


def flat_to_canvas(flat: torch.Tensor) -> torch.Tensor:
    """(B, NUM_PARAMS) -> raw canvases (B, NUM_CHANNELS, CANVAS, CANVAS).

    Padded entries are zeros. Preserves the input dtype.
    """
    B = flat.shape[0]
    canvas = flat.new_zeros(B, NUM_CHANNELS, CANVAS, CANVAS)
    off = 0
    for c, shape in enumerate(PARAM_SHAPES):
        rows, cols = placed_shape(shape)
        n = rows * cols
        canvas[:, c, :rows, :cols] = flat[:, off : off + n].reshape(B, rows, cols)
        off += n
    return canvas


def canvas_to_flat(canvas: torch.Tensor) -> torch.Tensor:
    """(B, NUM_CHANNELS, CANVAS, CANVAS) -> (B, NUM_PARAMS)."""
    B = canvas.shape[0]
    out = canvas.new_zeros(B, NUM_PARAMS)
    off = 0
    for c, shape in enumerate(PARAM_SHAPES):
        rows, cols = placed_shape(shape)
        n = rows * cols
        out[:, off : off + n] = canvas[:, c, :rows, :cols].reshape(B, n)
        off += n
    return out


def mask_flat() -> torch.Tensor:
    """Validity mask in flat parameter space, shape (NUM_PARAMS,), float {0,1}."""
    return canvas_to_flat(canvas_mask().unsqueeze(0)).squeeze(0)


# ---------------------------------------------------------------------------
# Per-tensor normalization (corpus statistics)
# ---------------------------------------------------------------------------
@dataclass
class ParamStats:
    """Per-tensor mean/std of valid parameters over the corpus.

    `mean`/`std` have shape (NUM_CHANNELS,), broadcast over each canvas
    channel; invalid (padded) entries stay at zero in normalized space.
    """

    mean: torch.Tensor  # (NUM_CHANNELS,)
    std: torch.Tensor   # (NUM_CHANNELS,)

    @staticmethod
    def from_flat(flat: torch.Tensor) -> "ParamStats":
        """Compute stats from raw flat weights (N, NUM_PARAMS)."""
        canvas = flat_to_canvas(flat)
        mask = canvas_mask().expand_as(canvas)
        means, stds = [], []
        for c in range(NUM_CHANNELS):
            vals = canvas[:, c][mask[:, c].bool()]
            means.append(vals.mean())
            stds.append(vals.std().clamp_min(1e-6))
        return ParamStats(torch.stack(means), torch.stack(stds))

    def to_json(self) -> dict:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @staticmethod
    def from_json(d: dict) -> "ParamStats":
        return ParamStats(torch.tensor(d["mean"]), torch.tensor(d["std"]))


NORM_RANGE = 4.0  # normalized values are z-scores clamped to +-NORM_RANGE, scaled by it


def normalize_canvas(canvas: torch.Tensor, stats: ParamStats) -> torch.Tensor:
    """Raw canvas -> diffusion-space canvas in roughly [-1, 1]; padding stays zero."""
    mask = canvas_mask().to(canvas.device)
    mean = stats.mean.to(canvas)[None, :, None, None]
    std = stats.std.to(canvas)[None, :, None, None]
    z = ((canvas - mean) / std).clamp(-NORM_RANGE, NORM_RANGE) / NORM_RANGE
    return z * mask


def denormalize_canvas(canvas: torch.Tensor, stats: ParamStats) -> torch.Tensor:
    """Diffusion-space canvas -> raw canvas (padding forced to zero)."""
    mask = canvas_mask().to(canvas.device)
    mean = stats.mean.to(canvas)[None, :, None, None]
    std = stats.std.to(canvas)[None, :, None, None]
    z = canvas * mask * NORM_RANGE
    return z * std + mean


def controller_from_flat(flat: torch.Tensor) -> Controller:
    """Build a Controller from a flat raw parameter vector (NUM_PARAMS,)."""
    net = Controller()
    net.set_flat(flat)
    return net
