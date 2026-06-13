"""NF4 kernel and pipeline profiling harness.

Measures where time is spent in a single diffusion step to identify
whether we're compute-bound or bandwidth-bound, and which kernels
are the bottleneck.

Usage:
    python bench.py --profile step    # Profile one full DiT step
    python bench.py --profile kernel  # Micro-benchmark NF4 matmul kernels
    python bench.py --profile memory  # Measure bandwidth utilization
"""

from __future__ import annotations

import argparse
import sys
import time

sys.path.insert(0, "/Users/noahlyons/dev/mlx/python")
sys.path.insert(0, ".")

import mlx.core as mx
import numpy as np


def bench_kernel():
    """Micro-benchmark NF4 quantized_matmul at various sizes."""
    print("=== NF4 Kernel Micro-benchmarks ===\n")

    # Sizes from actual Ideogram4 layers
    configs = [
        ("QKV proj (4608→13824)", 4608, 13824),
        ("Output proj (4608→4608)", 4608, 4608),
        ("FFN w1 (4608→12288)", 4608, 12288),
        ("FFN w2 (12288→4608)", 12288, 4608),
        ("AdaLN (512→18432)", 512, 18432),
        ("LLM cond proj (53248→4608)", 53248, 4608),
    ]

    # Token counts
    token_counts = [256, 1024, 4096]

    print(f"{'Layer':<30} {'Tokens':>7} {'NF4 ms':>8} {'Affine ms':>10} {'Ratio':>7} {'GFLOPS':>8}")
    print("-" * 80)

    for name, K, N in configs:
        # Create NF4 quantized weight
        w_float = mx.random.normal((N, K))
        wq_nf4, scales_nf4 = mx.quantize(w_float, mode="nf4")
        wq_aff, scales_aff, biases_aff = mx.quantize(w_float, mode="affine")
        mx.eval(wq_nf4, scales_nf4, wq_aff, scales_aff, biases_aff)

        for M in token_counts:
            x = mx.random.normal((1, M, K))
            mx.eval(x)

            # Warmup
            for _ in range(2):
                y = mx.quantized_matmul(x, wq_nf4, scales_nf4, mode="nf4",
                                        group_size=64, bits=4)
                mx.eval(y)

            # NF4 timing
            n_iters = 5
            t0 = time.perf_counter()
            for _ in range(n_iters):
                y = mx.quantized_matmul(x, wq_nf4, scales_nf4, mode="nf4",
                                        group_size=64, bits=4)
                mx.eval(y)
            nf4_ms = (time.perf_counter() - t0) / n_iters * 1000

            # Affine timing
            for _ in range(2):
                y = mx.quantized_matmul(x, wq_aff, scales_aff, biases_aff,
                                        mode="affine", group_size=64, bits=4)
                mx.eval(y)

            t0 = time.perf_counter()
            for _ in range(n_iters):
                y = mx.quantized_matmul(x, wq_aff, scales_aff, biases_aff,
                                        mode="affine", group_size=64, bits=4)
                mx.eval(y)
            aff_ms = (time.perf_counter() - t0) / n_iters * 1000

            flops = 2 * M * K * N / 1e9
            gflops_nf4 = flops / (nf4_ms / 1000)

            ratio = nf4_ms / aff_ms if aff_ms > 0 else 0
            print(f"{name:<30} {M:>7} {nf4_ms:>7.2f}  {aff_ms:>9.2f}  {ratio:>6.2f}x {gflops_nf4:>7.1f}")

    # Bandwidth analysis
    print("\n=== Bandwidth Analysis ===\n")
    # M4 Max has ~546 GB/s memory bandwidth
    # At 4-bit, a (N, K) weight matrix = N*K/2 bytes
    # Plus scales: N * (K/64) * 4 bytes for NF4 (float32 scales)
    # Plus input: M*K*2 bytes (bf16) + output: M*N*4 bytes (float32)
    print(f"{'Layer':<30} {'Weight MB':>10} {'BW-limited ms':>14} {'Actual ms':>10} {'Util%':>7}")
    print("-" * 80)
    BW_GBS = 546  # M4 Max theoretical

    for name, K, N in configs:
        weight_bytes = N * K / 2  # 4-bit packed
        scale_bytes = N * (K // 64) * 4  # float32 scales
        total_bytes = weight_bytes + scale_bytes
        bw_limited_ms = total_bytes / (BW_GBS * 1e9) * 1000

        # Re-measure at M=1 (pure bandwidth test)
        x = mx.random.normal((1, 1, K))
        wq, sc = mx.quantize(mx.random.normal((N, K)), mode="nf4")
        mx.eval(x, wq, sc)
        for _ in range(3):
            y = mx.quantized_matmul(x, wq, sc, mode="nf4", group_size=64, bits=4)
            mx.eval(y)
        t0 = time.perf_counter()
        for _ in range(10):
            y = mx.quantized_matmul(x, wq, sc, mode="nf4", group_size=64, bits=4)
            mx.eval(y)
        actual_ms = (time.perf_counter() - t0) / 10 * 1000

        util = bw_limited_ms / actual_ms * 100 if actual_ms > 0 else 0
        print(f"{name:<30} {total_bytes/1e6:>9.2f}  {bw_limited_ms:>13.4f}  {actual_ms:>9.3f}  {util:>6.1f}%")


def bench_step():
    """Profile one full DiT forward pass, breaking down by component."""
    import os
    import glob
    from huggingface_hub import hf_hub_download
    from hf_auth import resolve_hf_token
    from transformer import Ideogram4Transformer
    from load_weights import load_nf4_transformer
    from pipeline import IMAGE_POSITION_OFFSET

    token = resolve_hf_token()

    print("=== Single Step Profile ===\n")
    print("Loading model...", flush=True)
    f = hf_hub_download("ideogram-ai/ideogram-4-nf4",
                        "unconditional_transformer/diffusion_pytorch_model.safetensors",
                        token=token)
    model = Ideogram4Transformer()
    load_nf4_transformer(f, model, verbose=False)
    print("  Loaded\n", flush=True)

    for num_tokens in [256, 1024, 4096]:
        grid_side = int(num_tokens ** 0.5)
        pos = np.stack([
            np.zeros(num_tokens, dtype=np.int32),
            np.repeat(np.arange(grid_side), grid_side),
            np.tile(np.arange(grid_side), grid_side),
        ], axis=1) + IMAGE_POSITION_OFFSET
        position_ids = mx.array(pos[None, :, :])
        segment_ids = mx.ones((1, num_tokens), dtype=mx.int32)
        indicator = mx.full((1, num_tokens), 2, dtype=mx.int32)
        llm_features = mx.zeros((1, num_tokens, 53248), dtype=mx.bfloat16)
        x = mx.random.normal((1, num_tokens, 128)).astype(mx.bfloat16)
        t = mx.array([0.5])
        mx.eval(x, llm_features, position_ids)

        # Warmup
        out = model(llm_features=llm_features, x=x, t=t,
                    position_ids=position_ids, segment_ids=segment_ids,
                    indicator=indicator)
        mx.eval(out)

        # Timed
        n_iters = 3
        t0 = time.perf_counter()
        for _ in range(n_iters):
            out = model(llm_features=llm_features, x=x, t=t,
                        position_ids=position_ids, segment_ids=segment_ids,
                        indicator=indicator)
            mx.eval(out)
        step_ms = (time.perf_counter() - t0) / n_iters * 1000

        # Per-component breakdown (approximate via eval barriers)
        # Input projection + conditioning
        t0 = time.perf_counter()
        for _ in range(n_iters):
            ind_long = indicator.astype(mx.int32)
            img_mask = (ind_long == 2).astype(mx.bfloat16)[..., None]
            x_proj = model.input_proj(x * img_mask) * img_mask
            t_cond = model.t_embedding(t)
            import mlx.nn as nn
            adaln = nn.silu(model.adaln_proj(t_cond[:, None, :]))
            cos, sin = model.rotary_emb(position_ids)
            mx.eval(x_proj, adaln, cos, sin)
        input_ms = (time.perf_counter() - t0) / n_iters * 1000

        # One transformer block
        h = x_proj + model.embed_image_indicator(mx.zeros((1, num_tokens), dtype=mx.int32))
        mx.eval(h)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            h_out = model.layers[0](h, segment_ids=segment_ids, cos=cos.astype(h.dtype),
                                     sin=sin.astype(h.dtype), adaln_input=adaln)
            mx.eval(h_out)
        block_ms = (time.perf_counter() - t0) / n_iters * 1000

        print(f"Tokens={num_tokens} ({grid_side}x{grid_side}):")
        print(f"  Full step:      {step_ms:>8.1f} ms")
        print(f"  Input/cond:     {input_ms:>8.1f} ms")
        print(f"  1 block:        {block_ms:>8.1f} ms")
        print(f"  34 blocks est:  {block_ms * 34:>8.1f} ms ({block_ms * 34 / step_ms * 100:.0f}% of step)")
        print(f"  Overhead:       {step_ms - block_ms * 34 - input_ms:>8.1f} ms")
        print()


def main():
    parser = argparse.ArgumentParser(description="NF4 profiling harness")
    parser.add_argument("--profile", choices=["kernel", "step", "memory", "all"],
                        default="all")
    args = parser.parse_args()

    if args.profile in ("kernel", "all"):
        bench_kernel()
        print()

    if args.profile in ("step", "all"):
        bench_step()


if __name__ == "__main__":
    main()
