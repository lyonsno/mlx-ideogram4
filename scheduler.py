"""Logit-normal schedule and Euler flow-matching sampler for Ideogram4."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx


@dataclass(frozen=True)
class LogitNormalSchedule:
    mean: float
    std: float = 1.0
    logsnr_min: float = -15.0
    logsnr_max: float = 18.0

    def __call__(self, t: mx.array) -> mx.array:
        """Map uniform [0,1] step to noise level via logit-normal."""
        t = t.astype(mx.float32)
        # ndtri (inverse normal CDF) — use erfinv approximation
        z = mx.sqrt(mx.array(2.0)) * _erfinv(2.0 * t - 1.0)
        y = self.mean + self.std * z
        t_ = mx.sigmoid(-y)  # 1 - sigmoid(y)
        t_min = 1.0 / (1 + math.exp(0.5 * self.logsnr_max))
        t_max = 1.0 / (1 + math.exp(0.5 * self.logsnr_min))
        return mx.clip(t_, t_min, t_max)


def _erfinv(x: mx.array) -> mx.array:
    """Approximate inverse error function (Winitzki, 2008)."""
    a = 0.147
    ln_term = mx.log(1.0 - x * x)
    b = 2.0 / (math.pi * a) + ln_term / 2.0
    return mx.sign(x) * mx.sqrt(mx.sqrt(b * b - ln_term / a) - b)


def get_schedule_for_resolution(
    image_resolution: tuple[int, int],
    known_resolution: tuple[int, int] = (512, 512),
    known_mean: float = 0.5,
    std: float = 1.0,
) -> LogitNormalSchedule:
    num_pixels = image_resolution[0] * image_resolution[1]
    known_pixels = known_resolution[0] * known_resolution[1]
    mean = known_mean + 0.5 * math.log(num_pixels / known_pixels)
    return LogitNormalSchedule(mean=mean, std=std)


def make_step_intervals(num_steps: int) -> mx.array:
    return mx.linspace(0.0, 1.0, num_steps + 1)
