# Probolē: Text Encoding Alignment

## Target

- `/Users/noahlyons/dev/mlx-ideogram4/pipeline.py` (build_inputs, sample)
- `/Users/noahlyons/dev/mlx-ideogram4/text_encoder.py` (hidden state extraction)
- The inline text encoding in the sampling test scripts

## Reference

- `/tmp/ideogram4_pipeline.py` lines 329-480 (_tokenize, _build_inputs, _get_qwen3_vl_embeddings, _encode_text)

## Review scope

Compare MLX text encoding pipeline against PyTorch reference at each boundary.
The model components work individually but spatial coherence is not emerging
after sampling, suggesting a misalignment at one of the integration seams.

## Review context mode

Target files + reference file. Focus on data flow discrepancies.

## What to check

### Boundary 1: Tokenization
- Reference uses `tokenizer.apply_chat_template(messages, add_generation_prompt=True)`
  then tokenizes the result. MLX uses raw tokenization without chat template.
- Does the chat template wrapping change the tokens meaningfully?
- The reference tokenizes the JSON prompt inside a chat message structure.

### Boundary 2: Text encoder position IDs
- Reference builds `position_ids_4d` from `text_position_ids[..., 0]` expanded to 4 dims,
  then splits into `text_position_ids` (1D) and `mrope_position_ids` (3D).
  See reference lines 424-426.
- MLX broadcasts `arange(L)` to `(3, B, L)` for all 3 MRoPE axes equally.
- Are these equivalent? The reference sends 4 axes (1 text + 3 MRoPE), while
  our model takes 3 axes. The Qwen3-VL MRoPE may use 4-axis position IDs.

### Boundary 3: Hidden state stacking
- Reference: `torch.stack(selected, dim=0)` → `(13, B, L, 4096)`
  → `permute(1, 2, 3, 0)` → `(B, L, 4096, 13)` → `reshape(B, L, -1)` → `(B, L, 53248)`
- MLX: same operation. But verify the final dimension order is [layer0_feat, layer0_feat, ..., layer12_feat]
  not [layer0_dim0, layer1_dim0, ..., layer12_dim0, layer0_dim1, ...].
- The permute (1,2,3,0) puts layer index LAST before reshape, so features are interleaved
  by layer: [dim0_layer0, dim0_layer1, ..., dim0_layer12, dim1_layer0, ...].
  This is the OPPOSITE of concatenation along feature dim.
  Verify the DiT's llm_cond_proj expects this interleaved layout.

### Boundary 4: Sequence packing / indicator alignment
- Reference packs as: `[left_pad (zeros)][text_tokens][image_tokens]`
  with left padding when text < max_text_tokens.
- Reference sets `indicator[pad] = 0`, `indicator[text] = 3`, `indicator[image] = 2`.
- MLX: no left padding, text at start, image after.
- Reference masks LLM features at non-LLM positions: `text_mask = (indicator == 3).unsqueeze(-1)`
- Check: does the segment_id assignment match? Reference uses `SEQUENCE_PADDING_INDICATOR = -1`
  for padding, `1` for real tokens.

### Boundary 5: Sampling loop
- Reference iterates `i = num_steps-1 down to 0`.
- `t_val = schedule(step_intervals[i+1])`, `s_val = schedule(step_intervals[i])`
- `delta = s_val - t_val`
- `z = z + v * delta`
- Verify our schedule implementation matches. The erfinv approximation should be
  checked against scipy.special.ndtri for the operating range.
- Reference guidance schedule is `(7.0,)*45 + (3.0,)*3` for 48 steps,
  in REVERSE order (index 0 = last step, index 47 = first step).
  We use constant guidance. This shouldn't prevent coherence but may affect quality.

## Expected outcome

Identify the specific boundary where the MLX data flow diverges from reference.
The most likely candidates (in order of suspicion):
1. Hidden state stacking order (permute puts layers interleaved, not concatenated)
2. Text encoder position ID format (4-axis vs 3-axis)
3. Missing chat template tokenization
