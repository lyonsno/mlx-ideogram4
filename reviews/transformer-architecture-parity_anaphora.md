# Anaphora: Transformer Architecture Parity

**Probole**: `reviews/transformer-architecture-parity_probole.md`
**Reviewer**: cc-nf4-transformer-parity-review-0610 (Aposkepsis, no inherited context)
**Date**: 2026-06-10
**Target**: `transformer.py` (MLX)
**Reference**: `/tmp/ideogram4_model.py` (PyTorch)
**Verdict**: **No material structural parity defects found.**

---

## Module-by-module review

### 1. MRoPE interleaving — PASS

**Reference** (lines 96-104): Starts with temporal frequencies for all dims, then
scatters H-axis freqs at indices `1, 4, 7, ..., 58` (= `arange(1, 60, 3)`) and
W-axis freqs at indices `2, 5, 8, ..., 59` within the first
`mrope_section[axis]*3` positions. Critically, the scatter reads from the *same*
index positions in the source axis: `freqs[axis][..., idx]`.

**MLX** (lines 68-111): Precomputes an `axis_map` array where `axis_map[i]`
records which of the 3 axes to pull from for frequency dimension `i`. The gather
at lines 100-108 implements `out[bl, i] = freqs[axis_map[i], bl, i]`, which is
semantically identical to the reference's in-place scatter.

**Verified**: With `head_dim=256`, `half_dim=128`, `mrope_section=(24,20,20)`:
- H: length=60, indices 1..58 step 3 — all < 128.
- W: length=60, indices 2..59 step 3 — all < 128.
- Remaining dims 60-127 stay temporal (axis 0).
- The `emb = concat(freqs_out, freqs_out)` duplicates to full head_dim, matching
  reference line 103.

### 2. Attention — PASS

| Aspect | Reference | MLX | Match |
|--------|-----------|-----|-------|
| QKV projection | `Linear(4608, 13824, bias=False)` | Same | Yes |
| QKV split | `.view(B,L,3,H,D).unbind(2)` | `.reshape(B,L,3,H,D)` then slice `[:,:,i,:,:]` | Yes |
| QK-norm | `RMSNorm(head_dim, eps=1e-5)` before transpose | Same | Yes |
| Layout for SDPA | `(B,H,L,D)` via transpose(1,2) | `(B,H,L,D)` via transpose(0,2,1,3) | Yes |
| RoPE application | After QK-norm + transpose | Same | Yes |
| Segment mask | `(seg[:,None,:] == seg[:,:,None])[:,None,:,:]` boolean | Same shape, converted to additive `{0, -1e9}` | Yes (MLX SDPA requires additive mask) |
| Scale | Default `1/sqrt(d_k)` | Explicit `1.0/sqrt(head_dim)` | Yes |
| Output projection | `Linear(4608, 4608, bias=False)` | Same | Yes |

### 3. AdaLN modulation — PASS

**Chunk order** (both): `[scale_msa | gate_msa | scale_mlp | gate_mlp]`

| Aspect | Reference | MLX | Match |
|--------|-----------|-----|-------|
| Projection | `Linear(512, 4*4608, bias=True)` | Same | Yes |
| Gates | `tanh` | `tanh` | Yes |
| Scales | `1.0 + raw` | `1.0 + raw` | Yes |
| Attn residual | `x + gate_msa * norm2(attn_out)` | Same | Yes |
| FFN residual | `x + gate_mlp * norm2(ff(norm1(x) * scale_mlp))` | Same | Yes |
| Pre-norm with scale | `norm1(x) * scale` before sublayer | Same | Yes |
| Post-norm with gate | `gate * norm2(sublayer_out)` after sublayer | Same | Yes |

### 4. EmbedScalar / timestep embedding — PASS

| Aspect | Reference | MLX | Match |
|--------|-----------|-----|-------|
| Input scaling | `1e4 * (x - min) / (max - min)` | Same | Yes |
| Sinusoidal formula | `exp(arange(half) * -log(scale)/(half-1))` | Same | Yes |
| Sin/cos order | `cat(sin, cos)` | Same | Yes |
| MLP | `silu(Linear(D,D))` then `Linear(D,D)` | Same | Yes |
| Dtype after embed | `param_dtype` (via compute_dtype or weight.dtype) | Hardcoded `bfloat16` | **Intentional divergence** |

The bfloat16 hardcoding is correct for NF4 quantization where `weight.dtype`
would be `uint32`. The probole confirms this is expected.

**Minor note**: Reference pads odd-dim sinusoidal embeddings (line 228). MLX
omits this. Not a defect since `dim=4608` is always even, but a completeness gap
for hypothetical odd-dim configs.

### 5. FinalLayer — PASS

| Aspect | Reference | MLX | Match |
|--------|-----------|-----|-------|
| Norm type | `LayerNorm(eps=1e-6, elementwise_affine=False)` | `mx.fast.layer_norm(x, None, None, 1e-6)` | Yes |
| SiLU on conditioning | Inside: `silu(c)` before adaln projection | Same | Yes |
| Scale | `1.0 + adaln_modulation(silu(c))` | Same | Yes |
| Output projection | `Linear(4608, 128, bias=True)` | Same | Yes |

MLX inlines the layer_norm call instead of storing `norm_final` as a module.
Correct since `elementwise_affine=False` means no learnable parameters.

### 6. Forward pass / token masking — PASS

| Aspect | Reference | MLX | Match |
|--------|-----------|-----|-------|
| LLM mask | `(indicator == 3).float().unsqueeze(-1)` | `.astype(x.dtype)[..., None]` | Yes |
| Image mask | `(indicator == 2).float().unsqueeze(-1)` | Same pattern | Yes |
| LLM features masking | `llm_features * llm_mask` | Same | Yes |
| Image token masking | `x * img_mask` | Same | Yes |
| Input proj masking | `input_proj(x) * img_mask` | Same | Yes |
| t_cond unsqueeze | `t_cond.unsqueeze(1)` when t is per-sample | `t_cond[:, None, :]` | Yes |
| adaln_input | `F.silu(adaln_proj(t_cond))` | `nn.silu(adaln_proj(t_cond))` | Yes |
| LLM cond order | norm → proj → mask | Same | Yes |
| Image indicator embed | `embed((indicator==2).long())` | `embed((ind==2).astype(int32))` | Yes |
| Output dtype | `.to(torch.float32)` | `.astype(mx.float32)` | Yes |

### 7. Epsilon values — PASS

| Norm | Reference eps | MLX eps | Match |
|------|--------------|---------|-------|
| QK-norm (RMSNorm) | 1e-5 | 1e-5 | Yes |
| Block norms (attention_norm1/2, ffn_norm1/2) | config.norm_eps = 1e-5 | Same | Yes |
| llm_cond_norm | 1e-6 | 1e-6 | Yes |
| FinalLayer LayerNorm | 1e-6 | 1e-6 | Yes |
| RMSNorm class default | 1e-6 | 1e-6 | Yes |

### 8. Config — PASS

All default config values match between implementations:
- `emb_dim=4608`, `num_layers=34`, `num_heads=18`, `intermediate_size=12288`,
  `adanln_dim=512`, `in_channels=128`, `rope_theta=5_000_000`,
  `mrope_section=(24,20,20)`, `norm_eps=1e-5`
- `llm_features_dim = 4096 * 13 = 53248` (13 Qwen3-VL layers)

---

## Non-material notes

1. **Odd-dim sinusoidal padding**: Reference pads when `dim % 2 == 1`; MLX
   omits. Non-issue for default config (`dim=4608`).

2. **Attention mask value**: MLX uses `-1e9` as the masking penalty. This is
   sufficient even in bfloat16 (representable range extends to ~3.4e38). No
   risk of undermasking.

3. **QWEN3_VL_ACTIVATION_LAYERS**: MLX hardcodes
   `(0,3,6,9,12,15,18,21,24,27,30,33,35)` inline. Cannot verify against the
   reference's imported constant without seeing `ideogram4.constants`, but the
   count (13) and arithmetic (`4096*13=53248`) match the config.

4. **Initial dtype cast**: Reference casts `x`, `t`, `llm_features` to
   `param_dtype` at forward entry. MLX omits this cast, relying on MLX's
   operator-level dtype promotion. Correct for NF4 where `param_dtype` would
   resolve to uint32.

---

## Disposition

No material findings. No structural parity defects found across all 7 review
areas named in the probole. The MLX port is a faithful structural translation
of the PyTorch reference.
