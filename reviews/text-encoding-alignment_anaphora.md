# Anaphora: Text Encoding Alignment

**Reviewer:** cc-nf4-text-encoding-review-0611
**Probole:** reviews/text-encoding-alignment_probole.md
**Date:** 2026-06-11

---

## Summary

One critical bug found. The hidden state stacking order in `text_encoder.py`
produces a completely wrong feature layout compared to the reference. This alone
is sufficient to explain why the sampling loop produces incoherent latents
despite individual components working.

Two additional secondary findings: a schedule mean offset and a tokenization
ambiguity.

---

## Finding 1 — CRITICAL: Hidden state stacking order is wrong

**Files:** `text_encoder.py:146`, reference `_encode_text` lines 472-474

**Reference behavior:**
```python
stacked = torch.stack(selected, dim=0)      # (13, B, L, 4096)
stacked = torch.permute(stacked, (1,2,3,0)) # (B, L, 4096, 13)
stacked = stacked.reshape(B, L, -1)         # (B, L, 53248)
```

This produces an **interleaved** layout. For each token position, the 53248-dim
vector is ordered:

```
[dim0_layer0, dim0_layer1, ..., dim0_layer12,
 dim1_layer0, dim1_layer1, ..., dim1_layer12,
 ...
 dim4095_layer0, ..., dim4095_layer12]
```

**MLX behavior:**
```python
return mx.concatenate(features, axis=-1)  # [(B,L,4096)] * 13 → (B,L,53248)
```

This produces a **sequential/block** layout:

```
[dim0_layer0, dim1_layer0, ..., dim4095_layer0,   ← all of layer 0
 dim0_layer1, dim1_layer1, ..., dim4095_layer1,   ← all of layer 1
 ...
 dim0_layer12, ..., dim4095_layer12]              ← all of layer 12
```

**These are not equivalent.** The DiT's `llm_cond_proj` (a linear projection from
53248 → emb_dim) was trained on the interleaved layout. Feeding it the
sequential layout scrambles every feature dimension, producing effectively
random conditioning for the DiT.

**Fix:** Replace the concatenation with the equivalent stack+transpose+reshape:

```python
stacked = mx.stack(features, axis=0)           # (13, B, L, 4096)
stacked = mx.transpose(stacked, (1, 2, 3, 0))  # (B, L, 4096, 13)
B, L = input_ids.shape
return stacked.reshape(B, L, -1)               # (B, L, 53248)
```

**Severity:** This is the root cause of incoherent sampling output.

---

## Finding 2 — MODERATE: Schedule mean offset

**Files:** `scheduler.py:41`, reference `__call__` default `mu=0.5`

The MLX `get_schedule_for_resolution()` defaults `known_mean=1.0`.
The reference pipeline defaults `mu=0.5`, which is passed as `known_mean`.

For 1024×1024 generation:
- Reference mean: 0.5 + 0.5·ln(4) ≈ **1.193**
- MLX mean: 1.0 + 0.5·ln(4) ≈ **1.693**

This shifts the entire noise schedule toward higher noise levels. While it won't
prevent spatial coherence (the model can still denoise), it will produce
noticeably different quality/style than the reference and may cause the model to
operate outside its trained noise distribution.

**Fix:** Change default `known_mean=1.0` to `known_mean=0.5` in
`get_schedule_for_resolution()`, or ensure the caller passes `known_mean=0.5`.
Alternatively, if the reference's default `mu` is actually 0.5 per their API,
match it.

**Note:** The reference `__call__` also defaults `num_steps=128` while MLX
`sample()` defaults `num_steps=48`. This is a quality knob, not a correctness
bug.

---

## Finding 3 — LOW: Tokenization chat template ambiguity

**Files:** `text_encoder.py` (external caller responsibility)

The reference wraps prompts in a chat template before tokenizing:
```python
messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
```

The MLX `extract_text_features()` takes raw `input_ids`, and tokenization happens
externally. If the caller tokenizes without the chat template, the token sequence
will differ (missing system/role tokens, different BOS/EOS handling).

**Status:** Cannot confirm whether this is an active bug without seeing the
caller code. If the caller applies the chat template, this is a non-issue. If
not, the token sequence will be wrong.

---

## Non-findings (verified correct)

### Boundary 2: Position IDs — No bug

The reference splits a `(4, B, L)` position_ids tensor into:
- `text_position_ids = position_ids_4d[0]` → `(B, L)` for causal mask
- `mrope_position_ids = position_ids_4d[1:]` → `(3, B, L)` for RoPE

For text-only input, all 4 axes are identical sequential positions, so the
MRoPE effectively receives `(3, B, L)` with all axes equal.

The MLX text_encoder builds `position_ids = (3, B, L)` with all axes equal
(via `mx.broadcast_to(mx.arange(L).reshape(1,1,-1), (3,B,L))`), and passes this
through `LanguageModel.__call__` → `Qwen3VLModel.__call__` → each layer's
`Attention` → `Qwen3VLRotaryEmbedding.__call__`. The rotary embedding correctly
handles the `(3, B, L)` input at line 52-56 of `language.py`.

**The 4-axis vs 3-axis difference is a red herring.** The 4th axis in the
reference is only used for the causal mask position computation, not for RoPE.
Both paths send equivalent `(3, B, L)` positions to the rotary embedding.

### Boundary 4: Sequence packing — No bug for batch=1

For single-prompt inference (batch=1), `max_text_tokens = num_text_tokens`, so
`pad_len = 0` in the reference. The MLX layout `[text][image]` matches the
reference layout `[pad=0][text][image]`.

The `segment_ids` difference (MLX sets all to 1; reference sets padding to -1)
is moot when there's no padding.

**Note:** For batch>1 with different prompt lengths, the MLX code would need
left-padding support to match the reference.

### Boundary 5: erfinv approximation — No bug

The Winitzki (2008) erfinv approximation has max 0.02% relative error and max
2.8e-4 absolute schedule error across the full [0.04, 0.96] operating range.
The sampling loop structure (reverse iteration, delta computation, Euler update)
matches the reference exactly.

---

## Recommended fix priority

1. **Fix hidden state stacking order** (Finding 1) — this is the root cause
2. **Fix schedule mean default** (Finding 2) — needed for reference-matching output
3. **Document chat template requirement** (Finding 3) — verify caller code
