"""Attention breakdown profiler for Ideogram4 transformer blocks.

Measures time spent in each component of a transformer block:
QKV projection, QK-norm, RoPE, attention mask, SDPA, output proj,
FFN (w1/w3/w2), AdaLN modulation, norms.
"""

import sys
import time

sys.path.insert(0, "/Users/noahlyons/dev/mlx/python")
sys.path.insert(0, ".")

import math
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from transformer import (
    Ideogram4Transformer,
    Ideogram4Config,
    TransformerBlock,
    Attention,
    MLP,
    MRoPE,
    RMSNorm,
    _apply_rotary_pos_emb,
)
from pipeline import IMAGE_POSITION_OFFSET


def time_fn(fn, n_iters=5, warmup=2):
    """Time a function, returning ms per call."""
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    return (time.perf_counter() - t0) / n_iters * 1000


def bench_block_breakdown(block, num_tokens, head_dim=256, num_heads=18):
    """Break down one transformer block into components."""
    hidden = num_heads * head_dim  # 4608

    # Inputs
    x = mx.random.normal((1, num_tokens, hidden)).astype(mx.bfloat16)
    segment_ids = mx.ones((1, num_tokens), dtype=mx.int32)
    adaln_input = mx.random.normal((1, 1, 512)).astype(mx.bfloat16)
    mx.eval(x, segment_ids, adaln_input)

    # Pre-compute RoPE (shared across blocks in practice)
    grid_side = int(num_tokens ** 0.5)
    pos = np.stack([
        np.zeros(num_tokens, dtype=np.int32),
        np.repeat(np.arange(grid_side), grid_side),
        np.tile(np.arange(grid_side), grid_side),
    ], axis=1) + IMAGE_POSITION_OFFSET
    position_ids = mx.array(pos[None, :, :])

    config = Ideogram4Config()
    mrope = MRoPE(head_dim=head_dim, base=config.rope_theta, mrope_section=config.mrope_section)
    cos, sin = mrope(position_ids)
    cos = cos.astype(mx.bfloat16)
    sin = sin.astype(mx.bfloat16)
    mx.eval(cos, sin)

    results = {}

    # === Full block ===
    def full_block():
        out = block(x, segment_ids=segment_ids, cos=cos, sin=sin, adaln_input=adaln_input)
        mx.eval(out)
    results["Full block"] = time_fn(full_block)

    # === AdaLN modulation ===
    def adaln_mod():
        mod = block.adaln_modulation(adaln_input)
        mx.eval(mod)
    results["AdaLN modulation"] = time_fn(adaln_mod)

    # === Pre-attention norm + scale ===
    def pre_attn_norm():
        mod = block.adaln_modulation(adaln_input)
        C = mod.shape[-1] // 4
        scale_msa = 1.0 + mod[..., :C]
        h = block.attention_norm1(x) * scale_msa
        mx.eval(h)
    results["Pre-attn norm+scale"] = time_fn(pre_attn_norm)

    # === QKV projection ===
    attn = block.attention
    def qkv_proj():
        qkv = attn.qkv(x)
        mx.eval(qkv)
    results["QKV proj"] = time_fn(qkv_proj)

    # === QK norm ===
    def qk_norm():
        qkv = attn.qkv(x)
        qkv_r = qkv.reshape(1, num_tokens, 3, num_heads, head_dim)
        q = qkv_r[:, :, 0, :, :]
        k = qkv_r[:, :, 1, :, :]
        q = attn.norm_q(q)
        k = attn.norm_k(k)
        mx.eval(q, k)
    results["QK norm"] = time_fn(qk_norm)

    # === RoPE application ===
    q_test = mx.random.normal((1, num_heads, num_tokens, head_dim)).astype(mx.bfloat16)
    k_test = mx.random.normal((1, num_heads, num_tokens, head_dim)).astype(mx.bfloat16)
    mx.eval(q_test, k_test)
    def rope_apply():
        qr, kr = _apply_rotary_pos_emb(q_test, k_test, cos, sin)
        mx.eval(qr, kr)
    results["RoPE apply"] = time_fn(rope_apply)

    # === Segment mask construction ===
    def build_mask():
        mask = mx.expand_dims(segment_ids, 2) == mx.expand_dims(segment_ids, 1)
        mask = mx.expand_dims(mask, 1)
        mask = mx.where(mask, mx.array(0.0, dtype=mx.bfloat16),
                        mx.array(-1e9, dtype=mx.bfloat16))
        mx.eval(mask)
    results["Segment mask"] = time_fn(build_mask)

    # === SDPA only ===
    v_test = mx.random.normal((1, num_heads, num_tokens, head_dim)).astype(mx.bfloat16)
    mask_test = mx.zeros((1, 1, num_tokens, num_tokens), dtype=mx.bfloat16)
    mx.eval(v_test, mask_test)
    def sdpa():
        out = mx.fast.scaled_dot_product_attention(
            q_test, k_test, v_test,
            scale=1.0 / math.sqrt(head_dim),
            mask=mask_test,
        )
        mx.eval(out)
    results["SDPA"] = time_fn(sdpa)

    # === SDPA without mask ===
    def sdpa_nomask():
        out = mx.fast.scaled_dot_product_attention(
            q_test, k_test, v_test,
            scale=1.0 / math.sqrt(head_dim),
        )
        mx.eval(out)
    results["SDPA (no mask)"] = time_fn(sdpa_nomask)

    # === Output projection ===
    attn_out = mx.random.normal((1, num_tokens, hidden)).astype(mx.bfloat16)
    mx.eval(attn_out)
    def out_proj():
        o = attn.o(attn_out)
        mx.eval(o)
    results["Output proj"] = time_fn(out_proj)

    # === Post-attn norm + gate ===
    def post_attn():
        h = block.attention_norm2(attn_out)
        mx.eval(h)
    results["Post-attn norm"] = time_fn(post_attn)

    # === FFN (w1 + w3 + silu + w2) ===
    def ffn():
        out = block.feed_forward(x)
        mx.eval(out)
    results["FFN (w1+w3+silu+w2)"] = time_fn(ffn)

    # === FFN breakdown ===
    def ffn_w1():
        out = block.feed_forward.w1(x)
        mx.eval(out)
    results["  FFN w1"] = time_fn(ffn_w1)

    def ffn_w3():
        out = block.feed_forward.w3(x)
        mx.eval(out)
    results["  FFN w3"] = time_fn(ffn_w3)

    def ffn_w2():
        h = mx.random.normal((1, num_tokens, 12288)).astype(mx.bfloat16)
        mx.eval(h)
        out = block.feed_forward.w2(h)
        mx.eval(out)
    results["  FFN w2"] = time_fn(ffn_w2)

    return results


def main():
    import os
    import glob
    from huggingface_hub import hf_hub_download
    from load_weights import load_nf4_transformer

    token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()

    print("Loading model...", flush=True)
    f = hf_hub_download("ideogram-ai/ideogram-4-nf4",
                        "unconditional_transformer/diffusion_pytorch_model.safetensors",
                        token=token)
    model = Ideogram4Transformer()
    load_nf4_transformer(f, model, verbose=False)
    print("Loaded\n", flush=True)

    block = model.layers[0]

    for num_tokens in [256, 1024, 4096]:
        print(f"{'=' * 60}")
        print(f"  {num_tokens} tokens ({int(num_tokens**0.5)}x{int(num_tokens**0.5)})")
        print(f"{'=' * 60}")

        results = bench_block_breakdown(block, num_tokens)

        full = results["Full block"]
        print(f"\n{'Component':<25} {'ms':>8} {'% of block':>12}")
        print("-" * 48)
        for name, ms in results.items():
            pct = ms / full * 100
            bar = "█" * int(pct / 2)
            print(f"{name:<25} {ms:>7.2f}  {pct:>5.1f}% {bar}")
        print()


if __name__ == "__main__":
    main()
