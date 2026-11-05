"""Compact text-conditioned UNet over the weight canvas (SD-style).

Input:  noisy weight canvas (6ch) concatenated with the static validity mask
        (6ch), 32x32. Conditioning: timestep embedding (FiLM in every residual
        block) and text token embeddings via cross-attention at 16x16 and 8x8.
Output: v-prediction (6ch).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DiffusionConfig


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half) / half)
    args = t.float().unsqueeze(-1) * freqs
    return torch.cat([args.cos(), args.sin()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch * 2)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.time_mlp(F.silu(temb)).chunk(2, dim=-1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(h))
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention + text cross-attention + FFN, pre-norm, zero-init outputs."""

    def __init__(self, ch: int, heads: int, text_dim: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm1 = nn.GroupNorm(8, ch)
        self.to_qkv = nn.Linear(ch, 3 * ch)
        self.proj1 = nn.Linear(ch, ch)
        nn.init.zeros_(self.proj1.weight)
        nn.init.zeros_(self.proj1.bias)

        self.norm2 = nn.GroupNorm(8, ch)
        self.to_q = nn.Linear(ch, ch)
        self.to_kv = nn.Linear(text_dim, 2 * ch)
        self.proj2 = nn.Linear(ch, ch)
        nn.init.zeros_(self.proj2.weight)
        nn.init.zeros_(self.proj2.bias)

        self.norm3 = nn.GroupNorm(8, ch)
        self.ff = nn.Sequential(nn.Linear(ch, ch * 4), nn.GELU(), nn.Linear(ch * 4, ch))
        nn.init.zeros_(self.ff[-1].weight)
        nn.init.zeros_(self.ff[-1].bias)

    def forward(self, x: torch.Tensor, text: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)                    # (B, N, C)

        # Self-attention
        h = self.norm1(x).flatten(2).transpose(1, 2)
        q, k, v = self.to_qkv(h).chunk(3, dim=-1)
        q, k, v = (t.view(B, -1, self.heads, C // self.heads).transpose(1, 2) for t in (q, k, v))
        attended = F.scaled_dot_product_attention(q, k, v)
        tokens = tokens + self.proj1(attended.transpose(1, 2).reshape(B, -1, C))

        # Cross-attention to text tokens; bool mask True = attend.
        h = self.norm2(x).flatten(2).transpose(1, 2)
        q = self.to_q(h).view(B, -1, self.heads, C // self.heads).transpose(1, 2)
        k, v = self.to_kv(text).chunk(2, dim=-1)
        k = k.view(B, -1, self.heads, C // self.heads).transpose(1, 2)
        v = v.view(B, -1, self.heads, C // self.heads).transpose(1, 2)
        key_mask = text_mask.bool()[:, None, None, :]            # (B, 1, 1, L)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask)
        tokens = tokens + self.proj2(attended.transpose(1, 2).reshape(B, -1, C))

        # FFN
        h = self.norm3(x).flatten(2).transpose(1, 2)
        tokens = tokens + self.ff(h)

        return tokens.transpose(1, 2).reshape(B, C, H, W)


class _Level(nn.ModuleDict):
    """One resolution level: optional down/up sampler, then res blocks (+ attention)."""

    def __init__(self, blocks: list[nn.Module], sampler: nn.Module | None, name: str) -> None:
        items: dict[str, nn.Module] = {}
        if sampler is not None:
            items[name] = sampler  # "up" runs first; "down" runs last (see forward)
        for i, b in enumerate(blocks):
            items[f"block_{i}"] = b
        super().__init__(items)


class UNet(nn.Module):
    def __init__(self, cfg: DiffusionConfig) -> None:
        super().__init__()
        self.cfg = cfg
        chs = [cfg.base_channels * m for m in cfg.channel_mults]   # e.g. [48, 96, 192]
        time_dim = cfg.time_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim)
        )
        self.conv_in = nn.Conv2d(cfg.in_channels, chs[0], 3, padding=1)

        # Down path: at each level, res blocks (+attn), save skip, then downsample.
        self.down_levels = nn.ModuleList()
        ch = chs[0]
        res = 32
        for level, out_ch in enumerate(chs):
            blocks: list[nn.Module] = []
            for _ in range(cfg.res_blocks_per_level):
                blocks.append(ResBlock(ch, out_ch, time_dim))
                ch = out_ch
            if res in cfg.attn_resolutions:
                blocks.append(AttentionBlock(ch, cfg.num_heads, cfg.text_dim))
            sampler = None
            if level < len(chs) - 1:
                sampler = nn.Conv2d(ch, ch, 3, stride=2, padding=1)
                res //= 2
            self.down_levels.append(_Level(blocks, sampler, "down"))

        # Mid
        self.mid1 = ResBlock(ch, ch, time_dim)
        self.mid_attn = AttentionBlock(ch, cfg.num_heads, cfg.text_dim)
        self.mid2 = ResBlock(ch, ch, time_dim)

        # Up path, levels in reverse: upsample (except first), concat skip, blocks.
        self.up_levels = nn.ModuleList()
        res = 8
        for level in reversed(range(len(chs))):
            out_ch = chs[level]
            sampler = None
            if level < len(chs) - 1:
                sampler = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(ch, ch, 3, padding=1),
                )
                res *= 2
            blocks = []
            for i in range(cfg.res_blocks_per_level):
                in_ch = (out_ch + ch) if i == 0 else out_ch
                blocks.append(ResBlock(in_ch, out_ch, time_dim))
                ch = out_ch
            if res in cfg.attn_resolutions:
                blocks.append(AttentionBlock(ch, cfg.num_heads, cfg.text_dim))
            self.up_levels.append(_Level(blocks, sampler, "up"))

        self.final_norm = nn.GroupNorm(8, ch)
        self.final_conv = nn.Conv2d(ch, cfg.out_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        """x: (B, in_channels, 32, 32); t: (B,); text: (B, L, D); text_mask: (B, L)."""
        temb = self.time_mlp(timestep_embedding(t, self.cfg.time_dim))
        h = self.conv_in(x)

        skips: list[torch.Tensor] = []
        for level in self.down_levels:
            for key, layer in level.items():
                if key == "down":
                    continue
                if isinstance(layer, ResBlock):
                    h = layer(h, temb)
                else:
                    h = layer(h, text, text_mask)
            skips.append(h)  # every level's output is a skip for the up path
            if "down" in level:
                h = level["down"](h)

        h = self.mid1(h, temb)
        h = self.mid_attn(h, text, text_mask)
        h = self.mid2(h, temb)

        for level in self.up_levels:
            if "up" in level:
                h = level["up"](h)
            first = True
            for key, layer in level.items():
                if key == "up":
                    continue
                if isinstance(layer, ResBlock):
                    if first:  # concat this level's skip into the first block
                        h = torch.cat([h, skips.pop()], dim=1)
                        first = False
                    h = layer(h, temb)
                else:
                    h = layer(h, text, text_mask)

        return self.final_conv(F.silu(self.final_norm(h)))
