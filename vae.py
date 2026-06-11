"""Flux2 KL autoencoder decoder for Ideogram4, ported to MLX.

Only the decoder is needed for inference (latents → pixels).
Architecture: ResNet blocks + mid attention + progressive upsampling.

All convolutions use MLX's NHWC layout internally. PyTorch NCHW weights
are transposed during loading. Input/output use NCHW for compatibility
with the pipeline's unpatchify.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


def swish(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def _nchw_to_nhwc(x: mx.array) -> mx.array:
    return x.transpose(0, 2, 3, 1)


def _nhwc_to_nchw(x: mx.array) -> mx.array:
    return x.transpose(0, 3, 1, 2)


class GroupNorm32(nn.Module):
    """GroupNorm with 32 groups. Input/output: NHWC."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((1, 1, 1, num_channels))
        self.bias = mx.zeros((1, 1, 1, num_channels))
        self.num_groups = 32
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, H, W, C) — NHWC
        B, H, W, C = x.shape
        G = self.num_groups
        x = x.reshape(B, H, W, G, C // G)
        mean = mx.mean(x, axis=(1, 2, 4), keepdims=True)
        var = mx.var(x, axis=(1, 2, 4), keepdims=True)
        x = (x - mean) / mx.sqrt(var + self.eps)
        x = x.reshape(B, H, W, C)
        return x * self.weight + self.bias


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm1 = GroupNorm32(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = GroupNorm32(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.nin_shortcut = None

    def __call__(self, x: mx.array) -> mx.array:
        # x: NHWC
        h = swish(self.norm1(x))
        h = self.conv1(h)
        h = swish(self.norm2(h))
        h = self.conv2(h)
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.norm = GroupNorm32(in_channels)
        # Diffusers stores these as Linear, not Conv2d
        self.q = nn.Linear(in_channels, in_channels)
        self.k = nn.Linear(in_channels, in_channels)
        self.v = nn.Linear(in_channels, in_channels)
        self.proj_out = nn.Linear(in_channels, in_channels)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, H, W, C) NHWC
        B, H, W, C = x.shape
        h = self.norm(x)
        h = h.reshape(B, H * W, C)

        q = self.q(h)[:, None, :, :]  # (B, 1, HW, C)
        k = self.k(h)[:, None, :, :]
        v = self.v(h)[:, None, :, :]

        h = mx.fast.scaled_dot_product_attention(q, k, v, scale=C ** -0.5)
        h = self.proj_out(h[:, 0, :, :]).reshape(B, H, W, C)

        return x + h


class Upsample(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, H, W, C) NHWC
        B, H, W, C = x.shape
        x = mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)  # upsample H and W
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, resnets: list[ResnetBlock], upsample: Upsample | None = None):
        super().__init__()
        self.resnets = resnets
        self.upsamplers = [upsample] if upsample else []

    def __call__(self, h: mx.array) -> mx.array:
        for block in self.resnets:
            h = block(h)
        for up in self.upsamplers:
            h = up(h)
        return h


class Decoder(nn.Module):
    """Flux2 KL-VAE decoder: z_channels=32 → 3-channel RGB."""

    def __init__(
        self,
        ch: int = 128,
        out_ch: int = 3,
        ch_mult: list[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        z_channels: int = 32,
    ):
        super().__init__()
        num_resolutions = len(ch_mult)

        # post_quant_conv: 1x1 conv on z
        self.post_quant_conv = nn.Conv2d(z_channels, z_channels, kernel_size=1)

        # z → block_in
        block_in = ch * ch_mult[-1]  # 512
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # Middle
        self.mid_block_1 = ResnetBlock(block_in, block_in)
        self.mid_attn_1 = AttnBlock(block_in)
        self.mid_block_2 = ResnetBlock(block_in, block_in)

        # Upsampling blocks — match diffusers ordering (0=coarsest, N-1=finest)
        self.up_blocks = []
        for i_level in reversed(range(num_resolutions)):
            block_out = ch * ch_mult[i_level]
            resnets = []
            for _ in range(num_res_blocks + 1):
                resnets.append(ResnetBlock(block_in, block_out))
                block_in = block_out
            up = UpBlock(resnets, Upsample(block_in) if i_level != 0 else None)
            self.up_blocks.append(up)  # append in coarsest→finest order

        # Output
        self.norm_out = GroupNorm32(block_in)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def __call__(self, z: mx.array) -> mx.array:
        # Input: NCHW from pipeline. Convert to NHWC for MLX convolutions.
        z = _nchw_to_nhwc(z)

        z = self.post_quant_conv(z)
        h = self.conv_in(z)

        # Middle
        h = self.mid_block_1(h)
        h = self.mid_attn_1(h)
        h = self.mid_block_2(h)

        # Upsampling (0=coarsest → N-1=finest)
        for up_block in self.up_blocks:
            h = up_block(h)

        # Output
        h = swish(self.norm_out(h))
        h = self.conv_out(h)

        # Back to NCHW for pipeline
        return _nhwc_to_nchw(h)


def decode_latents(
    decoder: Decoder,
    z: mx.array,
    grid_h: int,
    grid_w: int,
    latent_shift: mx.array,
    latent_scale: mx.array,
    patch_size: int = 2,
) -> mx.array:
    """Decode latent tokens to pixel image.

    Args:
        z: (B, num_tokens, 128) latent tokens
        grid_h, grid_w: spatial grid dimensions
        latent_shift, latent_scale: (128,) normalization constants
        patch_size: 2 for Flux2

    Returns:
        (B, 3, H, W) pixel image in [0, 255] uint8
    """
    B = z.shape[0]
    ae_channels = z.shape[-1] // (patch_size * patch_size)  # 32

    # Denormalize
    z = z * latent_scale + latent_shift

    # Unpatchify: (B, grid_h*grid_w, 128) → (B, 32, grid_h*2, grid_w*2)
    z = z.reshape(B, grid_h, grid_w, patch_size, patch_size, ae_channels)
    z = z.transpose(0, 5, 1, 3, 2, 4)  # (B, ae_ch, grid_h, patch, grid_w, patch)
    z = z.reshape(B, ae_channels, grid_h * patch_size, grid_w * patch_size)

    # Decode
    decoded = decoder(z.astype(mx.float32))

    # To uint8 image
    decoded = mx.clip(decoded, -1.0, 1.0)
    decoded = ((decoded + 1.0) * 127.5).astype(mx.uint8)

    return decoded
