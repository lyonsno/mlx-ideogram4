"""Load Ideogram4 NF4 weights from HuggingFace into the MLX transformer."""

from __future__ import annotations

import json
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from safetensors import safe_open


def _repack_bnb_nf4_to_mlx(packed_uint8: np.ndarray, orig_shape: list[int]) -> np.ndarray:
    """Convert bitsandbytes NF4 packed uint8 to MLX uint32 format."""
    flat = packed_uint8.ravel()
    lo = flat & 0x0F
    hi = (flat >> 4) & 0x0F
    # bnb stores: high nibble = first element, low nibble = second element
    indices = np.stack([hi, lo], axis=-1).ravel().astype(np.uint32)
    rows, cols = orig_shape
    indices_grouped = indices.reshape(-1, 8)
    shifts = np.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=np.uint32)
    packed_u32 = np.sum(indices_grouped << shifts, axis=1).astype(np.uint32)
    return packed_u32.reshape(rows, cols // 8)


def load_nf4_transformer(
    safetensors_path: str,
    model: nn.Module,
    verbose: bool = True,
) -> None:
    """Load NF4 quantized weights into an Ideogram4Transformer.

    Swaps nn.Linear layers with nn.QuantizedLinear(mode='nf4') for
    quantized weights, and loads non-quantized tensors directly.
    """
    t0 = time.perf_counter()

    # === Pass 1: Load all weights from safetensors ===
    quantized_layers = {}  # base_name -> (weight_uint32, scales_float32, orig_shape)
    regular_tensors = {}

    with safe_open(safetensors_path, framework="numpy") as sf:
        keys = set(sf.keys())

        # Find quantized weight bases
        quantized_bases = set()
        for k in keys:
            if ".quant_state.bitsandbytes__nf4" in k:
                base = k.split(".quant_state.bitsandbytes__nf4")[0]
                quantized_bases.add(base)

        for base in sorted(quantized_bases):
            meta_key = f"{base}.quant_state.bitsandbytes__nf4"
            absmax_key = f"{base}.absmax"
            if meta_key not in keys or absmax_key not in keys:
                continue

            meta = json.loads(sf.get_tensor(meta_key).tobytes().decode("utf-8"))
            orig_shape = meta["shape"]
            blocksize = meta.get("blocksize", 64)

            if meta.get("nested", False):
                raise NotImplementedError(
                    f"Double quantization (nested absmax) not supported for {base}. "
                    "Re-quantize with compress_statistics=False."
                )

            packed = sf.get_tensor(base)
            absmax = sf.get_tensor(absmax_key)

            rows, cols = orig_shape
            mlx_packed = _repack_bnb_nf4_to_mlx(packed, orig_shape)
            n_groups = cols // blocksize
            absmax_reshaped = absmax.reshape(rows, n_groups)

            quantized_layers[base] = (
                mx.array(mlx_packed),
                mx.array(absmax_reshaped),
                orig_shape,
                blocksize,
            )

        # Build consumed set
        consumed = set()
        for base in quantized_bases:
            consumed.add(base)
            for suffix in [".absmax", ".quant_map", ".nested_absmax",
                           ".nested_quant_map", ".nested_scale_offset",
                           ".quant_state.bitsandbytes__nf4"]:
                consumed.add(base + suffix)

        # Load numpy-compatible tensors
        for k in sorted(keys - consumed):
            try:
                regular_tensors[k] = mx.array(sf.get_tensor(k))
            except (TypeError, ValueError):
                pass

    # Load bfloat16 tensors via MLX native loader
    remaining = sorted((keys - consumed) - set(regular_tensors.keys()))
    if remaining:
        bf16_weights = mx.load(safetensors_path)
        for k in remaining:
            if k in bf16_weights:
                regular_tensors[k] = bf16_weights[k]

    if verbose:
        print(f"  Parsed {len(quantized_layers)} quantized + {len(regular_tensors)} "
              f"regular tensors ({time.perf_counter() - t0:.1f}s)")

    # === Pass 2: Swap Linear -> QuantizedLinear and load weights ===

    # Build the weight dict that model.load_weights expects
    # For quantized layers: the safetensors key is like "layers.0.attention.qkv.weight"
    # We need to set up QuantizedLinear at "layers.0.attention.qkv" with
    # weight=uint32, scales=float32
    weight_pairs = []

    for base, (wq, scales, orig_shape, blocksize) in quantized_layers.items():
        # base is like "layers.0.attention.qkv.weight" — strip .weight
        if base.endswith(".weight"):
            param_path = base
            module_path = base[:-7]
        else:
            param_path = base
            module_path = base

        # Swap the nn.Linear at module_path with nn.QuantizedLinear
        out_dims, in_dims_packed = wq.shape
        in_dims = in_dims_packed * 8  # 8 nibbles per uint32

        _swap_to_quantized_linear(model, module_path, in_dims, out_dims, blocksize)

        # Set the weight and scales
        weight_pairs.append((f"{module_path}.weight", wq))
        weight_pairs.append((f"{module_path}.scales", scales))

        # Check for bias in regular tensors
        bias_key = f"{module_path}.bias"
        if bias_key in regular_tensors:
            weight_pairs.append((f"{module_path}.bias", regular_tensors[bias_key]))
            del regular_tensors[bias_key]

    # Add remaining regular tensors
    for k, v in regular_tensors.items():
        weight_pairs.append((k, v))

    # Load all weights
    model.load_weights(weight_pairs, strict=False)

    if verbose:
        print(f"  Loaded {len(weight_pairs)} weight entries into model "
              f"({time.perf_counter() - t0:.1f}s)")


def _swap_to_quantized_linear(
    model: nn.Module,
    path: str,
    in_dims: int,
    out_dims: int,
    group_size: int = 64,
) -> None:
    """Replace nn.Linear at the given dotted path with nn.QuantizedLinear(mode='nf4')."""
    parts = path.split(".")
    parent = model
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)

    leaf = parts[-1]
    old = getattr(parent, leaf) if not leaf.isdigit() else parent[int(leaf)]

    has_bias = isinstance(old, nn.Linear) and hasattr(old, "bias") and old.bias is not None

    ql = nn.QuantizedLinear(
        in_dims, out_dims, bias=has_bias, group_size=group_size, bits=4, mode="nf4"
    )

    if leaf.isdigit():
        parent[int(leaf)] = ql
    else:
        setattr(parent, leaf, ql)
