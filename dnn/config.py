"""Central configuration for diffusionNN.

Every size is fixed (canonical controller, weight canvas, UNet), mirroring
a text2image model with a fixed image resolution. Named presets select
CPU-friendly trade-offs between speed and quality.
"""

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Canonical controller ("the thing being generated")
# ---------------------------------------------------------------------------
IN_DIM = 8     # observations fed to the controller (tasks use the first k)
HID_DIM = 32   # hidden width of both hidden layers
OUT_DIM = 8    # controller output (tasks use the first m dims)

# ---------------------------------------------------------------------------
# Weight canvas ("the image" denoised by the diffusion model)
# ---------------------------------------------------------------------------
CANVAS = 32
NUM_CHANNELS = 6  # W1, b1, W2, b2, W3, b3 (each padded to CANVAS x CANVAS)

# Text conditioning
TEXT_DIM = 384   # sentence-transformers/all-MiniLM-L6-v2 hidden size
TEXT_LEN = 64    # cached token length per caption


@dataclass(frozen=True)
class DiffusionConfig:
    """UNet + diffusion-process hyper-parameters."""

    timesteps: int = 1000
    schedule: str = "cosine"
    schedule_s: float = 0.008
    prediction: str = "v"  # v-prediction

    in_channels: int = NUM_CHANNELS * 2  # noisy canvas + static mask channels
    out_channels: int = NUM_CHANNELS
    base_channels: int = 48
    channel_mults: tuple[int, ...] = (1, 2, 4)
    attn_resolutions: tuple[int, ...] = (16, 8)
    res_blocks_per_level: int = 2
    num_heads: int = 4
    time_dim: int = 256
    text_dim: int = TEXT_DIM
    text_len: int = TEXT_LEN
    cfg_dropout: float = 0.1  # probability of dropping the caption (CFG)


@dataclass(frozen=True)
class TrainConfig:
    """Training-loop hyper-parameters for the diffusion model."""

    steps: int = 15_000
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup: int = 500
    ema_decay: float = 0.999
    log_every: int = 50
    val_every: int = 500
    ckpt_every: int = 2_000
    val_batch_size: int = 128


@dataclass(frozen=True)
class Preset:
    """A named (model, training) configuration."""

    name: str
    model: DiffusionConfig
    train: TrainConfig


PRESETS: dict[str, Preset] = {
    # Validates the full pipeline end-to-end in minutes.
    "smoke": Preset(
        "smoke",
        DiffusionConfig(base_channels=24, channel_mults=(1, 2, 4), time_dim=128),
        TrainConfig(steps=800, batch_size=32, val_every=200, ckpt_every=400),
    ),
    # Primary CPU configuration (~a few hours).
    "small": Preset(
        "small",
        DiffusionConfig(),
        TrainConfig(steps=15_000),
    ),
    # Best quality, overnight CPU run.
    "base": Preset(
        "base",
        DiffusionConfig(base_channels=64, res_blocks_per_level=2),
        TrainConfig(steps=45_000, batch_size=64),
    ),
}


def get_preset(name: str) -> Preset:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Available: {sorted(PRESETS)}")
    return PRESETS[name]
