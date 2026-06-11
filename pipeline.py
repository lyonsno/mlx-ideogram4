"""Ideogram4 inference pipeline for MLX.

Minimal pipeline that runs the sampling loop with both transformers.
Text encoding uses the loaded Qwen3-VL with hidden state extraction.
VAE decode is a TODO — outputs raw latents for now.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from scheduler import LogitNormalSchedule, get_schedule_for_resolution, make_step_intervals

# Constants
LLM_TOKEN_INDICATOR = 3
OUTPUT_IMAGE_INDICATOR = 2
SEQUENCE_PADDING_INDICATOR = -1
IMAGE_POSITION_OFFSET = 65536
QWEN3_VL_ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)

# Latent normalization (from latent_norm.py)
LATENT_SHIFT = mx.array([
    0.01984364, 0.10149707, 0.29689495, 0.27188619, -0.21445648, -0.15979549,
    0.05021099, -0.15083604, -0.15360136, -0.20131799, 0.01922352, 0.0622626,
    0.10140969, -0.06739428, 0.3758261, -0.233712, 0.35164491, -0.02590912,
    -0.0271935, -0.10833897, -0.1476848, -0.01130957, -0.2298372, 0.23526423,
    -0.10893522, 0.11957631, 0.04047799, 0.3134589, -0.17225064, -0.18646109,
    -0.34691978, -0.03571246, 0.02583857, 0.10190072, 0.28402294, 0.26952152,
    -0.21634675, -0.17938656, 0.04358909, -0.15007621, -0.1548502, -0.18971131,
    0.02710861, 0.05609494, 0.10697846, -0.06854968, 0.38167698, -0.24269937,
    0.35705471, -0.03063305, -0.02946109, -0.11244286, -0.14336038, -0.01362137,
    -0.21863696, 0.23228983, -0.11739769, 0.11693044, 0.02563311, 0.31356594,
    -0.17420591, -0.19006285, -0.34905377, -0.04025005, 0.01924137, 0.07652984,
    0.2995608, 0.2628057, -0.22011674, -0.12715361, 0.04879879, -0.14075719,
    -0.15935895, -0.2123584, 0.01974813, 0.05523547, 0.10011992, -0.06428964,
    0.37781868, -0.21491644, 0.34254215, -0.03153528, -0.0310082, -0.10761415,
    -0.14730405, -0.02475182, -0.2285588, 0.2515081, -0.10445128, 0.12446,
    0.07062869, 0.30880162, -0.18016875, -0.18869164, -0.34533499, -0.0129177,
    0.02578168, 0.07993659, 0.28642181, 0.26038408, -0.22459419, -0.14820155,
    0.04059549, -0.14043529, -0.16111187, -0.2020305, 0.02602069, 0.04852717,
    0.10432153, -0.06309942, 0.38402443, -0.22397003, 0.34814481, -0.03774432,
    -0.03381438, -0.11245691, -0.14128767, -0.02853208, -0.21752016, 0.24872463,
    -0.11399775, 0.1222687, 0.05620835, 0.309178, -0.18065738, -0.19401479,
    -0.34495114, -0.01760592,
], dtype=mx.float32)

LATENT_SCALE = mx.array([
    1.63933691, 1.70204478, 1.73642566, 1.90004803, 1.6675316, 1.69059584,
    1.56853198, 1.62314944, 1.89106626, 1.58086668, 1.60822129, 1.60962993,
    1.63322129, 1.56074359, 1.73419528, 1.7919265, 1.64040632, 1.66802808,
    1.60390303, 1.75480492, 1.63187587, 1.64334594, 1.61722884, 1.60146046,
    1.63459219, 1.55291476, 1.68771497, 1.68415657, 1.78966054, 1.66631641,
    1.65626686, 1.65976433, 1.63487607, 1.69513249, 1.72933756, 1.91310663,
    1.67035057, 1.72286863, 1.56719251, 1.61934825, 1.88628859, 1.56911539,
    1.59455129, 1.60829869, 1.62470611, 1.56052853, 1.73677003, 1.77563606,
    1.63732541, 1.66370527, 1.59508952, 1.75153949, 1.63029275, 1.64517667,
    1.61659342, 1.59722044, 1.64103121, 1.5408531, 1.68610394, 1.67772755,
    1.78998563, 1.66621713, 1.65458955, 1.66041308, 1.64710857, 1.68163503,
    1.74000294, 1.92784786, 1.67411194, 1.67395548, 1.57406532, 1.62199356,
    1.87618195, 1.5584375, 1.57438785, 1.61711053, 1.63094305, 1.55644029,
    1.73124302, 1.80666627, 1.6463621, 1.65932006, 1.60816188, 1.75682671,
    1.64695873, 1.63121722, 1.61380832, 1.60478651, 1.63396035, 1.53505068,
    1.65534289, 1.67132281, 1.80317197, 1.6767314, 1.65700938, 1.68426259,
    1.65339716, 1.67540638, 1.73298504, 1.94067348, 1.67893609, 1.70635117,
    1.5730906, 1.61928553, 1.87148809, 1.56244866, 1.56697152, 1.61584394,
    1.62759496, 1.55480378, 1.73484107, 1.79055143, 1.64688773, 1.66121492,
    1.60135887, 1.75254572, 1.64798332, 1.62989921, 1.61381592, 1.60792883,
    1.63939668, 1.53075757, 1.65371318, 1.66801185, 1.80029087, 1.67591476,
    1.65655173, 1.68533454,
], dtype=mx.float32)


def build_inputs(
    num_text_tokens: int,
    height: int,
    width: int,
    patch_size: int = 2,
    ae_scale_factor: int = 8,
) -> dict:
    """Build the packed sequence layout for one prompt."""
    patch = patch_size * ae_scale_factor  # 16
    grid_h = height // patch
    grid_w = width // patch
    num_image_tokens = grid_h * grid_w
    total_seq_len = num_text_tokens + num_image_tokens

    # Image position IDs: (t=0, h, w) + offset
    h_idx = np.repeat(np.arange(grid_h), grid_w)
    w_idx = np.tile(np.arange(grid_w), grid_h)
    t_idx = np.zeros_like(h_idx)
    image_pos = np.stack([t_idx, h_idx, w_idx], axis=1) + IMAGE_POSITION_OFFSET

    # Text position IDs: (i, i, i) for sequential
    text_pos = np.stack([np.arange(num_text_tokens)] * 3, axis=1)

    position_ids = np.zeros((1, total_seq_len, 3), dtype=np.int32)
    position_ids[0, :num_text_tokens] = text_pos
    position_ids[0, num_text_tokens:] = image_pos

    segment_ids = np.full((1, total_seq_len), SEQUENCE_PADDING_INDICATOR, dtype=np.int32)
    segment_ids[0, :] = 1  # all tokens in same segment

    indicator = np.zeros((1, total_seq_len), dtype=np.int32)
    indicator[0, :num_text_tokens] = LLM_TOKEN_INDICATOR
    indicator[0, num_text_tokens:] = OUTPUT_IMAGE_INDICATOR

    return {
        "position_ids": mx.array(position_ids),
        "segment_ids": mx.array(segment_ids),
        "indicator": mx.array(indicator),
        "num_image_tokens": num_image_tokens,
        "num_text_tokens": num_text_tokens,
        "grid_h": grid_h,
        "grid_w": grid_w,
    }


def sample(
    cond_transformer: nn.Module,
    uncond_transformer: nn.Module,
    llm_features: mx.array,
    inputs: dict,
    *,
    num_steps: int = 48,
    guidance_scale: float = 7.0,
    height: int = 1024,
    width: int = 1024,
    seed: int = 42,
    verbose: bool = True,
) -> mx.array:
    """Run the Euler flow-matching sampling loop.

    Returns raw latents (B, num_image_tokens, 128) before VAE decode.
    """
    schedule = get_schedule_for_resolution((height, width))
    step_intervals = make_step_intervals(num_steps)

    num_image_tokens = inputs["num_image_tokens"]
    num_text_tokens = inputs["num_text_tokens"]
    latent_dim = 128

    mx.random.seed(seed)
    z = mx.random.normal((1, num_image_tokens, latent_dim))

    # Text padding for conditional transformer (zeros in text region)
    text_z_padding = mx.zeros((1, num_text_tokens, latent_dim))

    # Negative branch inputs (image-only, no text)
    neg_position_ids = inputs["position_ids"][:, num_text_tokens:]
    neg_segment_ids = inputs["segment_ids"][:, num_text_tokens:]
    neg_indicator = inputs["indicator"][:, num_text_tokens:]
    neg_llm_features = mx.zeros((1, num_image_tokens, llm_features.shape[-1]),
                                 dtype=llm_features.dtype)

    # Guidance schedule: constant for now
    gw = mx.array(guidance_scale)

    t0 = time.perf_counter()

    for i in range(num_steps - 1, -1, -1):
        t_val = schedule(step_intervals[i + 1: i + 2]).item()
        s_val = schedule(step_intervals[i: i + 1]).item()
        t = mx.array([[t_val]])

        # Conditional forward (text + image)
        pos_z = mx.concatenate([text_z_padding, z], axis=1)
        pos_out = cond_transformer(
            llm_features=llm_features,
            x=pos_z.astype(mx.bfloat16),
            t=t,
            position_ids=inputs["position_ids"],
            segment_ids=inputs["segment_ids"],
            indicator=inputs["indicator"],
        )
        pos_v = pos_out[:, num_text_tokens:]

        # Unconditional forward (image-only)
        neg_v = uncond_transformer(
            llm_features=neg_llm_features,
            x=z.astype(mx.bfloat16),
            t=t,
            position_ids=neg_position_ids,
            segment_ids=neg_segment_ids,
            indicator=neg_indicator,
        )

        # CFG blend
        v = gw * pos_v + (1.0 - gw) * neg_v
        delta = s_val - t_val
        z = z + v * delta

        mx.eval(z)

        if verbose and (i % 10 == 0 or i == num_steps - 1):
            elapsed = time.perf_counter() - t0
            print(f"  Step {num_steps - 1 - i + 1}/{num_steps}: "
                  f"t={t_val:.4f}→{s_val:.4f} "
                  f"v_norm={mx.sqrt(mx.mean(v * v)).item():.4f} "
                  f"({elapsed:.1f}s)")

    total = time.perf_counter() - t0
    if verbose:
        print(f"  Sampling complete: {num_steps} steps in {total:.1f}s "
              f"({total/num_steps:.2f}s/step)")

    return z
