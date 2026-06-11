"""Ideogram4 transformer backbone in MLX.

Port of the PyTorch Ideogram4Transformer to Apple Silicon via MLX,
supporting NF4 quantized weights for 4-bit inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

# Token role indicators
LLM_TOKEN_INDICATOR = 3
OUTPUT_IMAGE_INDICATOR = 2

# Qwen3-VL layers whose hidden states are concatenated
QWEN3_VL_ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)


@dataclass
class Ideogram4Config:
    emb_dim: int = 4608
    num_layers: int = 34
    num_heads: int = 18
    intermediate_size: int = 12288
    adanln_dim: int = 512
    in_channels: int = 128
    llm_features_dim: int = 4096 * len(QWEN3_VL_ACTIVATION_LAYERS)  # 53248
    rope_theta: int = 5_000_000
    mrope_section: tuple[int, ...] = (24, 20, 20)
    norm_eps: float = 1e-5


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


def _sinusoidal_embedding(t: mx.array, dim: int, scale: float = 1e4) -> mx.array:
    t = t.astype(mx.float32)
    half = dim // 2
    freq = math.log(scale) / (half - 1)
    freq = mx.exp(mx.arange(half, dtype=mx.float32) * -freq)
    emb = t[..., None] * freq
    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
    return emb


class MRoPE(nn.Module):
    """Multi-resolution rotary positional embeddings (3D: temporal, height, width)."""

    def __init__(self, head_dim: int, base: int, mrope_section: tuple[int, ...]):
        super().__init__()
        inv_freq = 1.0 / (
            base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim)
        )
        self.inv_freq = inv_freq
        self.mrope_section = tuple(mrope_section)
        self.head_dim = head_dim
        half_dim = head_dim // 2

        # Precompute the interleaving: for each position in the output freq
        # vector, which axis (0=temporal, 1=H, 2=W) does it come from?
        # Default: axis 0 (temporal). Override at positions idx%3==1 -> H,
        # idx%3==2 -> W, within the mrope_section range.
        axis_map = [0] * half_dim
        for axis, offset in ((1, 1), (2, 2)):
            length = mrope_section[axis] * 3
            for i in range(offset, length, 3):
                if i < half_dim:
                    axis_map[i] = axis
        self._axis_map = axis_map

    def __call__(self, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        # position_ids: (B, L, 3)
        B, L, _ = position_ids.shape
        half_dim = self.head_dim // 2

        # Compute per-axis frequencies: (3, B, L, D/2)
        pos = mx.transpose(position_ids, (2, 0, 1)).astype(mx.float32)  # (3, B, L)
        inv_freq = self.inv_freq[None, None, :, None]  # (1, 1, D/2, 1)

        # (3, B, D/2, L)
        freqs = mx.broadcast_to(inv_freq, (3, B, half_dim, 1)) @ pos[:, :, None, :]
        freqs = mx.transpose(freqs, (0, 1, 3, 2))  # (3, B, L, D/2)

        # Interleave: for each freq dim, pick from the right axis
        # freqs: (3, B, L, D/2), axis_map[i] tells which axis for dim i
        axis_idx = mx.array(self._axis_map, dtype=mx.int32)  # (D/2,)

        # Gather: freqs[axis_map[i], :, :, i] for each i
        # Reshape freqs to (3, B*L, D/2), gather axis 0, reshape back
        freqs_flat = freqs.reshape(3, B * L, half_dim)  # (3, BL, D/2)
        # We need: out[bl, i] = freqs_flat[axis_map[i], bl, i]
        # = freqs_flat transposed to (BL, D/2, 3), then index [..., axis_map]
        freqs_perm = mx.transpose(freqs_flat, (1, 2, 0))  # (BL, D/2, 3)
        freqs_out = mx.take_along_axis(
            freqs_perm, axis_idx[None, :, None], axis=2
        ).squeeze(2)  # (BL, D/2)
        freqs_out = freqs_out.reshape(B, L, half_dim)

        emb = mx.concatenate([freqs_out, freqs_out], axis=-1)  # (B, L, D)
        return mx.cos(emb), mx.sin(emb)


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([-x2, x1], axis=-1)


def _apply_rotary_pos_emb(
    q: mx.array, k: mx.array, cos: mx.array, sin: mx.array
) -> tuple[mx.array, mx.array]:
    # q, k: (B, H, L, D); cos, sin: (B, L, D)
    cos = cos[:, None, :, :]  # (B, 1, L, D)
    sin = sin[:, None, :, :]
    q_embed = q * cos + _rotate_half(q) * sin
    k_embed = k * cos + _rotate_half(k) * sin
    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, eps: float = 1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.norm_q = RMSNorm(self.head_dim, eps=eps)
        self.norm_k = RMSNorm(self.head_dim, eps=eps)
        self.o = nn.Linear(hidden_size, hidden_size, bias=False)

    def __call__(
        self,
        x: mx.array,
        segment_ids: mx.array,
        cos: mx.array,
        sin: mx.array,
    ) -> mx.array:
        B, L, _ = x.shape

        qkv = self.qkv(x)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0, :, :]  # (B, L, H, D)
        k = qkv[:, :, 1, :, :]
        v = qkv[:, :, 2, :, :]

        q = self.norm_q(q)
        k = self.norm_k(k)

        # (B, L, H, D) -> (B, H, L, D) for SDPA
        q = mx.transpose(q, (0, 2, 1, 3))
        k = mx.transpose(k, (0, 2, 1, 3))
        v = mx.transpose(v, (0, 2, 1, 3))

        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        # Segment mask: attend only within same segment
        # segment_ids: (B, L) -> mask: (B, 1, L, L)
        mask = mx.expand_dims(segment_ids, 2) == mx.expand_dims(segment_ids, 1)
        mask = mx.expand_dims(mask, 1)  # (B, 1, L, L)
        # Convert bool mask to additive mask for SDPA
        mask = mx.where(mask, mx.array(0.0, dtype=q.dtype), mx.array(-1e9, dtype=q.dtype))

        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=1.0 / math.sqrt(self.head_dim), mask=mask
        )
        out = mx.transpose(out, (0, 2, 1, 3)).reshape(B, L, self.hidden_size)
        return self.o(out)


class MLP(nn.Module):
    """SiLU-gated MLP (SwiGLU variant)."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        norm_eps: float,
        adanln_dim: int,
    ):
        super().__init__()
        self.attention = Attention(hidden_size, num_heads, eps=1e-5)
        self.feed_forward = MLP(hidden_size, intermediate_size)

        self.attention_norm1 = RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm1 = RMSNorm(hidden_size, eps=norm_eps)
        self.attention_norm2 = RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm2 = RMSNorm(hidden_size, eps=norm_eps)

        self.adaln_modulation = nn.Linear(adanln_dim, 4 * hidden_size, bias=True)

    def __call__(
        self,
        x: mx.array,
        segment_ids: mx.array,
        cos: mx.array,
        sin: mx.array,
        adaln_input: mx.array,
    ) -> mx.array:
        mod = self.adaln_modulation(adaln_input)
        # Split into 4 chunks along last dim
        C = mod.shape[-1] // 4
        scale_msa = 1.0 + mod[..., :C]
        gate_msa = mx.tanh(mod[..., C : 2 * C])
        scale_mlp = 1.0 + mod[..., 2 * C : 3 * C]
        gate_mlp = mx.tanh(mod[..., 3 * C :])

        attn_out = self.attention(
            self.attention_norm1(x) * scale_msa,
            segment_ids=segment_ids,
            cos=cos,
            sin=sin,
        )
        x = x + gate_msa * self.attention_norm2(attn_out)
        x = x + gate_mlp * self.ffn_norm2(
            self.feed_forward(self.ffn_norm1(x) * scale_mlp)
        )
        return x


class EmbedScalar(nn.Module):
    """Timestep embedding via sinusoidal + MLP."""

    def __init__(self, dim: int, input_range: tuple[float, float] = (0.0, 1.0)):
        super().__init__()
        self.dim = dim
        self.range_min = input_range[0]
        self.range_max = input_range[1]
        self.mlp_in = nn.Linear(dim, dim, bias=True)
        self.mlp_out = nn.Linear(dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = x.astype(mx.float32)
        scaled = 1e4 * (x - self.range_min) / (self.range_max - self.range_min)
        emb = _sinusoidal_embedding(scaled, self.dim)
        # Keep as bfloat16 for compute — don't cast to weight dtype
        # (weight may be uint32 when quantized)
        emb = emb.astype(mx.bfloat16)
        emb = nn.silu(self.mlp_in(emb))
        return self.mlp_out(emb)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, adanln_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaln_modulation = nn.Linear(adanln_dim, hidden_size, bias=True)
        self.hidden_size = hidden_size

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        scale = 1.0 + self.adaln_modulation(nn.silu(c))
        # LayerNorm without learnable affine
        x = mx.fast.layer_norm(x, None, None, 1e-6)
        return self.linear(x * scale)


class Ideogram4Transformer(nn.Module):
    """Ideogram 4 flow-matching diffusion transformer."""

    def __init__(self, config: Ideogram4Config | None = None):
        super().__init__()
        if config is None:
            config = Ideogram4Config()
        self.config = config

        head_dim = config.emb_dim // config.num_heads

        self.input_proj = nn.Linear(config.in_channels, config.emb_dim, bias=True)
        self.llm_cond_norm = RMSNorm(config.llm_features_dim, eps=1e-6)
        self.llm_cond_proj = nn.Linear(
            config.llm_features_dim, config.emb_dim, bias=True
        )
        self.t_embedding = EmbedScalar(config.emb_dim, input_range=(0.0, 1.0))
        self.adaln_proj = nn.Linear(config.emb_dim, config.adanln_dim, bias=True)

        self.embed_image_indicator = nn.Embedding(2, config.emb_dim)

        self.rotary_emb = MRoPE(
            head_dim=head_dim,
            base=config.rope_theta,
            mrope_section=config.mrope_section,
        )

        self.layers = [
            TransformerBlock(
                hidden_size=config.emb_dim,
                intermediate_size=config.intermediate_size,
                num_heads=config.num_heads,
                norm_eps=config.norm_eps,
                adanln_dim=config.adanln_dim,
            )
            for _ in range(config.num_layers)
        ]

        self.final_layer = FinalLayer(
            hidden_size=config.emb_dim,
            out_channels=config.in_channels,
            adanln_dim=config.adanln_dim,
        )

    def __call__(
        self,
        *,
        llm_features: mx.array,
        x: mx.array,
        t: mx.array,
        position_ids: mx.array,
        segment_ids: mx.array,
        indicator: mx.array,
    ) -> mx.array:
        """Velocity prediction.

        Args:
            llm_features: (B, L, 53248) Qwen3-VL hidden states from 13 layers.
            x: (B, L, 128) noisy image latent tokens.
            t: (B,) or (B, L) flow-matching timestep in [0, 1].
            position_ids: (B, L, 3) (temporal, height, width) for MRoPE.
            segment_ids: (B, L) segment IDs for packed-batch attention masking.
            indicator: (B, L) token role: LLM_TOKEN_INDICATOR(3) or OUTPUT_IMAGE_INDICATOR(2).

        Returns:
            (B, L, 128) velocity prediction. Only OUTPUT_IMAGE_INDICATOR positions
            are meaningful.
        """
        B, L, _ = x.shape

        indicator_long = indicator.astype(mx.int32)
        llm_mask = (indicator_long == LLM_TOKEN_INDICATOR).astype(x.dtype)[..., None]
        img_mask = (indicator_long == OUTPUT_IMAGE_INDICATOR).astype(x.dtype)[..., None]

        llm_features = llm_features * llm_mask
        x = x * img_mask

        x = self.input_proj(x) * img_mask

        # Timestep conditioning -> adaln modulation input
        t_cond = self.t_embedding(t)
        if t.ndim == 1:
            t_cond = t_cond[:, None, :]  # (B, 1, D)
        adaln_input = nn.silu(self.adaln_proj(t_cond))

        # Project LLM features
        llm_features = self.llm_cond_norm(llm_features)
        llm_features = self.llm_cond_proj(llm_features) * llm_mask

        h = x + llm_features

        # Image indicator embedding
        img_indicator = (indicator_long == OUTPUT_IMAGE_INDICATOR).astype(mx.int32)
        h = h + self.embed_image_indicator(img_indicator)

        # Positional encoding
        cos, sin = self.rotary_emb(position_ids)
        cos = cos.astype(h.dtype)
        sin = sin.astype(h.dtype)

        # Transformer blocks
        for layer in self.layers:
            h = layer(h, segment_ids=segment_ids, cos=cos, sin=sin, adaln_input=adaln_input)

        out = self.final_layer(h, c=adaln_input)
        return out.astype(mx.float32)
