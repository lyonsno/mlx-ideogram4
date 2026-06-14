# Anaphorá: Gradio Demo Launch Hardening

**Reviewer:** cc-nf4-demo-launch-review-0612 (Aposkepsis, fresh context)
**Probolē:** `reviews/gradio-demo-launch-hardening_probole.md`
**Date:** 2026-06-12
**Files reviewed:** `app.py`, `serve.sh`, `generate.py`, `README.md`, `pipeline.py` (for token math)

---

## Findings

### MATERIAL — must fix before Reddit link

#### S1. Gradio API bypasses all UI constraints (Security)

**File:** `app.py:444-447`

The `btn.click()` endpoint is exposed as a Gradio API endpoint by default (Gradio auto-generates `/api/predict` or `/call/...` routes). The `generate()` function does clamp width/height to 1024, but:

1. **Preset is not validated.** The `preset_name` arg is passed directly to `PRESETS[preset_name]` (line 174). An API caller can send an arbitrary string and get a `KeyError` crash, or — worse — if `PRESETS` were ever mutated to include more keys, they'd bypass any public-mode restriction. Currently this is just a crash (DoS, but single-request).

2. **No prompt length limit.** A long prompt (e.g., 100KB of text) flows through the tokenizer and then allocates a `(1, L, 53248)` bfloat16 tensor at line 224. At 100K tokens, that's `100000 × 53248 × 2 bytes ≈ 10 GB` — an OOM bomb from a single request. This is the most exploitable vector.

3. **`use_json=True` with a malformed JSON string** doesn't crash at line 172 (it goes through as raw), but does get wrapped in `messages` and tokenized. Not a crash vector itself, but worth noting that JSON mode passes user text through `json.dumps({"prompt": prompt_text})` only when `use_json=False` — when `True`, the raw string is passed, meaning an attacker could craft a `messages`-shaped payload that changes tokenizer behavior. Unlikely to be exploitable but worth a comment.

4. **The V4_QUALITY_48 preset (48 steps) is not locked out in public mode.** The `--public` help text says "turbo only" but nothing in the code restricts the dropdown to turbo. A 48-step generation at 1024×1024 would take ~25 minutes and hold the single queue slot the entire time.

**Recommendation:**
- Add `max_length` to the prompt textbox or truncate in `generate()` before tokenizing. A 2000-character cap is generous and prevents the OOM path.
- In `--public` mode, override the preset dropdown to `["V4_TURBO_12"]` and clamp resolution to 512.
- Validate `preset_name` with a `get()` + fallback instead of bare dict access.
- Consider `demo.launch(show_api=False)` to hide the API docs page (doesn't prevent API use but reduces discoverability).

#### S2. Rate limit is per-server, not per-user (Security)

**File:** `app.py:48-51`

`_last_gen_time` is a single global timestamp. If user A generates, user B is rate-limited for 30s even though A was the one who just generated. Conversely, this means the rate limit *does* protect against queue flooding — no one can generate faster than once per 30s regardless. But the UX is confusing: an innocent user arriving right after someone else's generation gets a rate-limit message with no explanation.

**Severity:** Low for security (it's actually more restrictive than intended), medium for UX. A per-IP or per-session rate limit would be better, but for a one-Mac demo, the global cooldown is defensible if documented.

**Recommendation:** Change the rate-limit message to say "The server is cooling down — please wait {wait}s (one Mac, shared cooldown)." No code change needed for security, just messaging.

#### S3. `trust_remote_code=True` on a gated model (Security)

**File:** `app.py:72`, `generate.py:141`

`AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)` executes arbitrary Python from the model repo. Since this is a gated HuggingFace model (ideogram-ai/ideogram-4-nf4), the risk is low — Ideogram controls the repo. But if the token leaked or the model repo were compromised, this is RCE on the serving Mac. Worth a comment in code at minimum; worth considering `trust_remote_code=False` if the tokenizer works without it (Qwen3 tokenizers usually do).

**Recommendation:** Test without `trust_remote_code=True`. If it works, remove it.

---

### NON-MATERIAL — should fix but not launch-blocking

#### S4. App.py and README performance numbers disagree (Claims)

**README table (lines 27-30):** 512×512 NF4 at 6.5s/step, 130s sampling, 11.5 GB peak.
**App.py table (lines 431-442):** 512×512 NF4 at 3.3s/step, 67s sampling, 11.5 GB peak.

These are 2× apart. The app.py table shows numbers matching the M4 Max 128 GB machine in an earlier commit message ("uncontended"). The README shows different numbers that are also labeled "uncontended, M4 Max 128 GB." One of them is wrong, or they were measured under different conditions (contended vs uncontended, different MLX builds, etc.).

The MFLUX column also differs: README says 8.9s/step, app.py says 3.3s/step. The app.py numbers look like they might be from a different machine or a different measurement epoch.

**This is the most dangerous claim issue.** If someone on Reddit benchmarks and gets 6.5s/step while the embedded table says 3.3s/step, it looks like overclaiming. Pick one set of numbers and make them consistent, or clearly label the conditions that differ.

**Recommendation:** Reconcile. If the app.py table is stale from a different machine/build, update it to match the README or remove it and link to the README.

#### S5. Removed provenance claim (Claims)

**File:** `app.py:441`, `README.md:149`

The README and app previously carried a compressed build-provenance marketing claim. That kind of claim attracts scrutiny on Reddit because the public commit history is visible. Not a code issue, but worth removing or verifying before going public.

#### S6. HuggingFace token read from file, not env (Security)

**File:** `app.py:59`, `generate.py:128`

```python
token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
```

This hardcodes the token path. It's the default HuggingFace location, but it means:
- If someone runs the demo in a Docker container or non-standard env, it crashes with an unhelpful `FileNotFoundError`.
- The `huggingface_hub` library itself reads this file; you could use `huggingface_hub.HfApi().token` or `huggingface_hub.utils.get_token()` instead, which handles multiple token sources.

Not a security issue per se (the token is already on disk), but fragile.

#### S7. No `--public` mode enforcement of resolution in UI (UX)

**File:** `app.py:396-397`

The sliders still show 256-1024 range in `--public` mode. The `generate()` function clamps to 1024, but the help text says "512 max." The code clamp is 1024, not 512. If public mode is supposed to cap at 512, the clamp at line 163 should use `512` not `1024`, and the slider `maximum` should be overridden.

If 1024 is intentionally allowed in public mode, the help text at line 455 is misleading.

#### S8. Gallery loads 13 images but shows `[:12]` (UX)

**File:** `app.py:423`

The gallery slices to 12 images. Currently 13 match the `nf4_*.png` glob. The 13th (likely `nf4_paint_512x512.png`) is silently dropped. Not a bug, but the cap is arbitrary. Consider raising to 16 or dropping the slice if all images are intentional gallery content.

#### S9. `serve.sh` uses `uv run` with stock MLX risk (Install)

**File:** `serve.sh:22-27`

The `--with "mlx @ git+..."` override in `uv run` depends on uv's resolution behavior. The README itself warns this can fail (`KeyError: 'nf4'`). If `serve.sh` is the LaunchAgent entry point for the public demo, and uv resolves stock MLX, the server starts and immediately crashes on first generation. The `_assert_nf4_available()` guard is in `generate.py` but *not* in `app.py`.

**Recommendation:** Add the NF4 probe from `generate.py:37-64` to `app.py` at module level (or at least in `_load_models()`). Currently, if the fork isn't installed, the first user to click Generate gets a cryptic `KeyError: 'nf4'` deep in model loading, not the helpful error message.

#### S10. Install: README six-command flow is actually five commands (Install)

**File:** `README.md:68-92`

The antechamber says "six commands from zero to generating." Counting the README install block: (1) git clone + cd, (2) pip install -e ., (3) pip install --force-reinstall, (4) huggingface-cli login, (5) python generate.py. That's five commands (or six if you count `cd` separately from `git clone`). Pedantic, but Reddit will count.

Also: step 4 requires visiting a URL to accept the license *before* `huggingface-cli login` works for downloading. The README mentions this but it's inside a comment, not a visible step. A stranger who skips the comment will get a 403 on first run and not know why.

#### S11. No generation timeout (Security)

**File:** `app.py:257-275`

There's no timeout on the sampling loop. A 48-step 1024×1024 generation takes ~25 minutes. If the server is in `--public` mode with queue size 1, one generation holds the entire server for 25 minutes. Combined with S1 (preset not locked to turbo), this is a low-cost DoS.

Even with turbo locked, 12 steps at 1024×1024 is still ~6 minutes per generation. Consider a hard timeout (e.g., 5 minutes) that kills the generation and returns an error.

---

### OBSERVATIONS — no fix needed

#### O1. LaunchAgent `--share --public` is reasonable

`serve.sh` passing `--share --public` to a LaunchAgent is fine for a demo box. The `--share` creates a Gradio tunnel (HTTPS, rate-limited by Gradio's infra), and `--public` enables the app-level restrictions. The main risk is the Mac being exposed to the internet, but that's inherent to the use case.

#### O2. Memory budget on 16 GB

With previews disabled (`NF4_NO_PREVIEW=1`), the sampling peak is ~11.5 GB. The VAE decode adds memory on top. On a 16 GB Mac, this leaves ~4.5 GB for the OS + VAE. The README's verified 16 GB run shows no swap. Defensible, but close — if macOS decides to cache something, the margin shrinks. The 512×512 clamp (if enforced per S7) keeps this safe since 1024×1024 peaks at 13.7 GB.

#### O3. Antechamber NF4 explainer is accurate

The `<details>` block in `app.py:338-349` correctly describes NF4, cites QLoRA, links bitsandbytes, and accurately states "Metal kernels for MLX." The "same checkpoint files, half the memory of FP8" claim is supported by the numbers (11.5 vs 28.1 GB).

#### O4. `_assert_nf4_available()` coverage

The guard in `generate.py` covers the main failure mode (stock MLX shadowing the fork). It probes actual quantization, not just import. It won't catch a partial fork install (e.g., Python bindings from fork but C++ from stock), but that's an unlikely failure mode.

#### O5. License visibility is good

README line 5 has the license callout as a blockquote. README line 172 repeats it. App.py line 370 links it in the antechamber. The non-commercial nature is visible in three places. Sufficient.

---

## Summary

| # | Severity | Category | Summary |
|---|----------|----------|---------|
| S1 | **MATERIAL** | Security | Unbounded prompt length → OOM; preset/resolution not locked in public mode; API crash on bad preset |
| S2 | Non-material | Security/UX | Global rate limit confusing to innocent users |
| S3 | **MATERIAL** | Security | `trust_remote_code=True` — test without it |
| S4 | Non-material | Claims | App.py and README performance numbers are 2× apart |
| S5 | Non-material | Claims | Build-provenance marketing claim — remove or verify against commit history |
| S6 | Non-material | Security | Fragile HF token path |
| S7 | Non-material | UX | Public mode help says "512 max" but code clamps to 1024 |
| S8 | Non-material | UX | Gallery drops 13th image silently |
| S9 | Non-material | Install | NF4 guard missing from app.py; serve.sh may start with stock MLX |
| S10 | Non-material | Install | "Six commands" is five; license-accept step is hidden |
| S11 | Non-material | Security | No generation timeout; 48-step 1024 holds server 25 min |

**Material findings: S1, S3.**
S1 is the real blocker — an unbounded prompt can OOM the server from the API. The fix is small (truncate prompt, lock preset/resolution in public mode, validate preset key).
S3 is worth testing before launch — if `trust_remote_code=False` works, it's a free security win.

Everything else is polish. The demo is well-structured, the NF4 explainer is accurate, the gallery works, and the core rate-limiting + queue design is sound for a single-Mac demo.

---

## Disposition: `cc/codex-gradio-launch-hardening-0614`

Recorded 2026-06-14 by Codex after applying the launch-hardening branch. Verification lives in `tests/test_app_hardening.py`; runtime evidence includes `evidence/live_runs/20260614T031410Z_nf4-mlx-metal_512x512_seed2025_smoke.json` and the matching PNG.

| # | Disposition | Linkage |
|---|---|---|
| S1 | Addressed | `app.py` now normalizes every request before model work: prompt capped at `MAX_PROMPT_CHARS`, invalid presets fall back, public mode allows only the 12-step and 20-step presets, public dimensions clamp to 1024, and launch hides Gradio API docs with `show_api=False`. Covered by `test_normalize_request_caps_prompt_bad_preset_and_public_limits`, `test_public_ui_limits_prompt_preset_and_resolution`, and `test_launch_kwargs_hide_api`. |
| S2 | Addressed as global-by-design | Cooldown remains shared-server, but the user-facing message now says the server is cooling down and names the shared cooldown. Covered by `test_cooldown_message_names_shared_server`. |
| S3 | Addressed | Tokenizer smoke passed with `trust_remote_code=False` (`Qwen2Tokenizer`, 15 tokens), and both `app.py` and `generate.py` now load the tokenizer without remote-code trust. Covered by `test_tokenizer_loads_without_remote_code_trust`. |
| S4 | Addressed by evidence-based claim correction | App and README no longer present the stale 3.3s value as the single current 512x512 headline. README and app now distinguish latest local smoke (`7.5s/step`, `151s`, `11.55 GB`) from older fast matrix receipts (`3.3-3.4s/step`, `67-69s`, `11.52 GB`). |
| S5 | Addressed by removal | The README no longer carries the build-provenance marketing section. Covered by `test_readme_does_not_carry_provenance_claim`. |
| S6 | Addressed | Both app and CLI now use `huggingface_hub.utils.get_token()` and fail with a clear login/license message instead of reading only `~/.cache/huggingface/token`. Covered by `test_token_helper_uses_huggingface_resolution` and `test_generate_uses_huggingface_token_helper`. |
| S7 | Addressed | Public mode UI now exposes the bounded public choices: 12-step or 20-step presets and 256-1024px sliders, matching the function-level clamp. Covered by `test_public_ui_limits_prompt_preset_and_resolution` and `test_public_copy_matches_public_caps_and_tone`. |
| S8 | Addressed | Gallery now shows all discovered NF4 images instead of slicing to 12. Covered by `test_gallery_displays_all_nf4_images`. |
| S9 | Addressed | `app.py` calls the existing `_assert_nf4_available()` guard before loading models. Covered indirectly by app source and real `uv run` NF4 probe (`nf4-probe-ok`). |
| S10 | Already addressed before branch | README and app install copy already used a five-command flow with visible license acceptance. |
| S11 | Partially addressed / bounded | Public mode now excludes the 48-step preset and caps requests at 20 steps and 1024px. A public-mode wall-clock timeout guard also exists (`NF4_PUBLIC_TIMEOUT_SECONDS`, default 900s). It cannot interrupt an in-flight single MLX kernel call mid-step, but it stops continuing after the budget trips between steps. |

Additional surfaced mismatch addressed: `docs/public-demo.md` documented `./serve.sh --public --tunnel ngrok`, while `serve.sh` did not implement it. The branch now parses `--tunnel ngrok`, requires `NGROK_DOMAIN`, starts the local Gradio app, waits for it to answer, and launches `ngrok http`; covered by `test_serve_script_implements_documented_ngrok_tunnel`.
