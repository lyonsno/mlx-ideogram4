"""Ideogram4 text encoder: Qwen3-VL with multi-layer hidden state extraction.

Uses mlx-vlm's Qwen3-VL architecture with the Ideogram4 NF4 weights.
Extracts hidden states from 13 intermediate layers and concatenates them
to produce 53248-dim features per token for the DiT.

The text encoder is a modified Qwen3-VL-8B with custom deepstack_merger_list
modules. The Ideogram4 NF4 weights include these custom modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

# Qwen3-VL activation layers whose hidden states feed the DiT
QWEN3_VL_ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)


def _patch_model_for_hidden_states(model_inner, target_layers: tuple[int, ...]):
    """Monkey-patch the Qwen3VLModel.__call__ to collect intermediate hidden states.

    Returns a callable that extracts the collected hidden states after forward.
    """
    original_call = model_inner.__class__.__call__
    collected = {}

    def patched_call(self, *args, **kwargs):
        # We need to intercept the layer loop. Since we can't easily
        # modify the loop without copying the whole method, we wrap each
        # target layer to save its output.
        nonlocal collected
        collected.clear()

        # Get inputs_embeds
        inputs = args[0] if args else kwargs.get("inputs")
        inputs_embeds = kwargs.get("inputs_embeds", None)
        if len(args) > 1:
            inputs_embeds = args[1]

        if inputs_embeds is None:
            h = self.embed_tokens(inputs)
        else:
            h = inputs_embeds

        cache = kwargs.get("cache", None)
        if len(args) > 3:
            cache = args[3]
        if cache is None:
            cache = [None] * len(self.layers)

        mask = kwargs.get("mask", None)
        if len(args) > 2:
            mask = args[2]
        if mask is None:
            from mlx_vlm.models.qwen3_vl.language import create_attention_mask
            mask = create_attention_mask(
                h, cache[0] if cache and cache[0] is not None else cache
            )

        position_ids = kwargs.get("position_ids", None)
        if len(args) > 4:
            position_ids = args[4]
        deepstack_visual_embeds = kwargs.get("deepstack_visual_embeds", None)
        visual_pos_masks = kwargs.get("visual_pos_masks", None)

        for layer_idx, (layer, c) in enumerate(zip(self.layers, cache)):
            h = layer(h, mask, c, position_ids)
            # Deepstack processing (same as original)
            if deepstack_visual_embeds is not None and layer_idx in range(
                len(deepstack_visual_embeds)
            ):
                h = self._deepstack_process(
                    h, visual_pos_masks, deepstack_visual_embeds[layer_idx]
                )
            # Collect hidden states at target layers
            if layer_idx in target_layers:
                collected[layer_idx] = h

        return self.norm(h)

    model_inner.__class__.__call__ = patched_call

    def get_collected():
        return collected

    return get_collected


def extract_text_features(
    model,
    input_ids: mx.array,
    target_layers: tuple[int, ...] = QWEN3_VL_ACTIVATION_LAYERS,
) -> mx.array:
    """Run the text encoder and extract multi-layer hidden states.

    Args:
        model: mlx-vlm Qwen3-VL Model instance
        input_ids: (B, L) token IDs
        target_layers: Which transformer layers to extract from

    Returns:
        (B, L, 53248) concatenated hidden states from target layers
        (4096 per layer × 13 layers = 53248)
    """
    # Patch model to collect hidden states
    inner_model = model.language_model.model
    get_collected = _patch_model_for_hidden_states(inner_model, set(target_layers))

    # Run forward pass (text-only, no images)
    inputs_embeds = inner_model.embed_tokens(input_ids)

    # Simple forward without cache or images
    cache = [None] * len(inner_model.layers)
    mask = None  # Will be created inside

    # Position IDs: simple sequential for text-only
    B, L = input_ids.shape
    position_ids = mx.broadcast_to(
        mx.arange(L).reshape(1, 1, -1),
        (3, B, L),  # 3 axes for MRoPE
    )

    _ = model.language_model(
        input_ids,
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
    )

    # Collect and concatenate hidden states
    collected = get_collected()
    features = []
    for layer_idx in sorted(target_layers):
        if layer_idx in collected:
            features.append(collected[layer_idx])
        else:
            raise RuntimeError(
                f"Layer {layer_idx} hidden states not collected. "
                f"Available: {sorted(collected.keys())}"
            )

    # Stack and interleave: match reference permute(1,2,3,0)+reshape layout
    # This produces [dim0_layer0, dim0_layer1, ..., dim0_layer12, dim1_layer0, ...]
    # NOT the sequential layout from concatenate
    stacked = mx.stack(features, axis=0)           # (13, B, L, 4096)
    stacked = mx.transpose(stacked, (1, 2, 3, 0))  # (B, L, 4096, 13)
    return mx.reshape(stacked, (B, L, -1))          # (B, L, 53248)
