"""Load Ideogram4 NF4 text encoder (modified Qwen3-VL) weights into mlx-vlm model."""

from __future__ import annotations

import json
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from safetensors import safe_open


def _repack_bnb_nf4_to_mlx(packed_uint8: np.ndarray, orig_shape: list[int]) -> np.ndarray:
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


def _sanitize_key(k: str) -> str:
    """Map bitsandbytes safetensors key to mlx-vlm model key.

    Safetensors:  language_model.layers.0.self_attn.q_proj.weight
    mlx-vlm:      language_model.model.layers.0.self_attn.q_proj.weight

    Safetensors:  visual.blocks.0.attn.qkv.weight
    mlx-vlm:      vision_tower.blocks.0.attn.qkv.weight
    """
    if k.startswith("language_model."):
        # language_model.X → language_model.model.X
        rest = k[len("language_model."):]
        # But NOT for language_model.norm (that stays as language_model.model.norm)
        return f"language_model.model.{rest}"
    elif k.startswith("visual."):
        return k.replace("visual.", "vision_tower.", 1)
    return k


def load_nf4_text_encoder(
    safetensors_path: str,
    model: nn.Module,
    verbose: bool = True,
) -> None:
    """Load NF4 quantized text encoder weights into mlx-vlm Qwen3-VL model."""
    t0 = time.perf_counter()

    quantized_layers = {}
    regular_tensors = {}

    with safe_open(safetensors_path, framework="numpy") as sf:
        keys = set(sf.keys())

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
            if meta.get("nested", False):
                raise NotImplementedError(f"Double quantization not supported for {base}")

            orig_shape = meta["shape"]
            blocksize = meta.get("blocksize", 64)

            packed = sf.get_tensor(base)
            absmax = sf.get_tensor(absmax_key)

            rows, cols = orig_shape

            # bitsandbytes flattens the weight and pads to blocksize.
            # If cols is not divisible by blocksize, we need to pad.
            cols_padded = ((cols + blocksize - 1) // blocksize) * blocksize
            if cols_padded != cols:
                # Unpack to indices, pad, then repack
                flat = packed.ravel()
                lo = flat & 0x0F
                hi = (flat >> 4) & 0x0F
                indices = np.stack([lo, hi], axis=-1).ravel().astype(np.uint32)
                # Pad to rows * cols_padded with zeros
                total_padded = rows * cols_padded
                if len(indices) < total_padded:
                    indices = np.pad(indices, (0, total_padded - len(indices)))
                else:
                    indices = indices[:total_padded]
                indices_grouped = indices.reshape(-1, 8)
                shifts = np.array([0, 4, 8, 12, 16, 20, 24, 28], dtype=np.uint32)
                mlx_packed = np.sum(indices_grouped << shifts, axis=1).astype(np.uint32)
                mlx_packed = mlx_packed.reshape(rows, cols_padded // 8)

                # Pad absmax similarly
                n_groups = cols_padded // blocksize
                total_groups = rows * n_groups
                if absmax.size < total_groups:
                    absmax = np.pad(absmax, (0, total_groups - absmax.size))
                absmax_reshaped = absmax[:total_groups].reshape(rows, n_groups)
                actual_cols = cols_padded
            else:
                mlx_packed = _repack_bnb_nf4_to_mlx(packed, orig_shape)
                n_groups = cols // blocksize
                absmax_reshaped = absmax.reshape(rows, n_groups)
                actual_cols = cols

            quantized_layers[base] = (
                mx.array(mlx_packed),
                mx.array(absmax_reshaped),
                [rows, actual_cols],
                blocksize,
            )

        consumed = set()
        for base in quantized_bases:
            consumed.add(base)
            for suffix in [".absmax", ".quant_map", ".nested_absmax",
                           ".nested_quant_map", ".nested_scale_offset",
                           ".quant_state.bitsandbytes__nf4"]:
                consumed.add(base + suffix)

        for k in sorted(keys - consumed):
            try:
                regular_tensors[k] = mx.array(sf.get_tensor(k))
            except (TypeError, ValueError):
                pass

    # bfloat16 fallback
    remaining = sorted((keys - consumed) - set(regular_tensors.keys()))
    if remaining:
        bf16_weights = mx.load(safetensors_path)
        for k in remaining:
            if k in bf16_weights:
                regular_tensors[k] = bf16_weights[k]

    if verbose:
        print(f"  Parsed {len(quantized_layers)} quantized + {len(regular_tensors)} "
              f"regular tensors ({time.perf_counter() - t0:.1f}s)")

    # Build weight pairs with key sanitization + QuantizedLinear swap
    weight_pairs = []

    for base, (wq, scales, orig_shape, blocksize) in quantized_layers.items():
        if base.endswith(".weight"):
            module_path = _sanitize_key(base[:-7])
        else:
            module_path = _sanitize_key(base)

        out_dims, in_dims_packed = wq.shape
        in_dims = in_dims_packed * 8

        _swap_to_quantized_linear(model, module_path, in_dims, out_dims, blocksize)

        weight_pairs.append((f"{module_path}.weight", wq))
        weight_pairs.append((f"{module_path}.scales", scales))

        bias_key = base.replace(".weight", ".bias") if base.endswith(".weight") else f"{base}.bias"
        sanitized_bias_key = _sanitize_key(bias_key)
        if bias_key in regular_tensors:
            weight_pairs.append((sanitized_bias_key, regular_tensors.pop(bias_key)))

    for k, v in regular_tensors.items():
        weight_pairs.append((_sanitize_key(k), v))

    model.load_weights(weight_pairs, strict=False)

    if verbose:
        print(f"  Loaded {len(weight_pairs)} weight entries ({time.perf_counter() - t0:.1f}s)")


def _swap_to_quantized_linear(model, path, in_dims, out_dims, group_size=64):
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

    ql = nn.QuantizedLinear(in_dims, out_dims, bias=has_bias, group_size=group_size, bits=4, mode="nf4")

    if leaf.isdigit():
        parent[int(leaf)] = ql
    else:
        setattr(parent, leaf, ql)
