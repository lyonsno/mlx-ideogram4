# Probolē: Transformer Architecture Parity

## Target

- `/Users/noahlyons/dev/mlx-ideogram4/transformer.py` (MLX implementation)

## Reference

- `/tmp/ideogram4_model.py` (PyTorch source, downloaded from
  https://github.com/ideogram-oss/ideogram4/main/src/ideogram4/modeling_ideogram4.py)

## Review scope

Structural parity between the MLX transformer port and the PyTorch reference.
This is a line-by-line architecture review, not a numerical parity test.

## Review context mode

Target file + reference file. No inherited implementation context.

## What to look for

1. **MRoPE implementation**: The interleaved frequency assignment is the most
   complex part. Reference uses `freqs_t[..., idx] = freqs[axis][..., idx]`
   with in-place scatter. MLX uses `take_along_axis` with a precomputed axis
   map. Verify the axis map construction matches the reference's interleaving
   pattern: temporal at indices 0,3,6,...; height at 1,4,7,...; width at 2,5,8,...
   within the first `mrope_section[axis]*3` positions.

2. **Attention**: Reference uses fused QKV `Linear(4608, 13824)` with
   `.view(B, L, 3, H, D).unbind(2)`. MLX must produce the same Q/K/V split.
   QK-norm (RMSNorm per head) must be applied before RoPE. Segment mask must
   be block-diagonal (same segment = attend).

3. **AdaLN modulation**: 4 chunks from `Linear(512, 4*4608)`. Gates use
   `tanh` (not sigmoid). Scales are `1 + output`. The gate applies AFTER
   the post-norm (norm2), not before. Verify the exact residual structure:
   `x = x + gate * norm2(sublayer_output)`.

4. **Timestep embedding**: `EmbedScalar` scales input to `[0, 1e4]`, applies
   sinusoidal embedding of dim=4608, then MLP. The `adaln_proj` projects
   4608 → 512 with SiLU. The resulting 512-dim vector is broadcast across
   all sequence positions (unsqueeze when t is per-sample).

5. **Final layer**: Uses `LayerNorm` (not RMSNorm), elementwise_affine=False.
   SiLU is applied to the conditioning INSIDE the final layer (not outside).
   Output is cast to float32.

6. **Token masking**: LLM features masked to indicator==3 positions, image
   tokens masked to indicator==2. Input projection output is also masked.
   Image indicator embedding uses `(indicator == 2).long()` as index (0 or 1).

7. **Forward pass dtype handling**: Reference casts inputs to `param_dtype`
   (the weight dtype). With NF4 quantization, the weight dtype is uint32,
   so the MLX implementation must NOT cast to weight dtype. Verify the
   bfloat16 hardcoding in EmbedScalar is correct.

## Out of scope

- Numerical parity testing (requires weight loading + reference comparison)
- Weight loading (separate review)
- Performance
