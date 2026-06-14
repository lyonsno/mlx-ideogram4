# Clean Pull Smoke Receipt: nlm2pr macOS 26 M2 Pro 16 GB

## Verdict

Pass. A fresh clone of `https://github.com/lyonsno/mlx-ideogram4.git` at `origin/main` installed in a fresh venv, replaced stock MLX with the NF4 fork installed last, passed an explicit NF4 quantization probe, and generated a 512x512 / 20-step image on this 16 GB M2 Pro Mac.

This receipt proves a clean repo/venv/install/runtime path on this box. It does not prove a cold Hugging Face model download, because `model/ideogram-ai/ideogram-4-nf4` was already present in the shared Hugging Face cache.

## Box

- Host: `nlm2pr.local`
- macOS: `26.5.1` build `25F80`
- Darwin: `25.5.0`
- Hardware: `Mac14,9`
- CPU/GPU family: Apple `M2 Pro`
- Unified memory: `17179869184` bytes

## Source

- Clone root: `/private/tmp/nlm2pr-macos26-m2pro16gb-clean-pull-20260614T021404Z`
- Evidence root: `/Users/noahlyons/dev/mlx-ideogram4/evidence/clean_smokes/nlm2pr-macos26-m2pro16gb-clean-pull-20260614T021404Z`
- Repo URL: `https://github.com/lyonsno/mlx-ideogram4.git`
- Checked out commit: `4cbf56e786bf912d5ce3b00d83791160fe4ece9d`

## Install Path Exercised

1. `git clone https://github.com/lyonsno/mlx-ideogram4.git`
2. `python3 -m venv .venv`
3. `.venv/bin/python -m pip install -e .`
4. `.venv/bin/python -m pip install --force-reinstall --no-deps git+https://github.com/lyonsno/mlx.git@nf4`

The editable install first pulled stock `mlx==0.31.2` and `mlx-metal==0.31.2` as transitive dependencies. The final fork reinstall replaced stock MLX with:

- Fork commit: `eccb857825439dd8c24c5cc5d27e9489fc2f4eef`
- Installed package: `mlx-0.32.0.dev20260613+eccb857`
- Built wheel: `mlx-0.32.0.dev20260613+eccb857-cp311-cp311-macosx_26_0_arm64.whl`
- Wheel SHA256 from pip log: `c9ca5dac9be4972c3cfce18b418b048c80d528e2e91d0e0f8d28b584d6cb7b98`

## NF4 Probe

Command shape:

```python
import mlx.core as mx
x = mx.zeros((64,64)).astype(mx.float16)
mx.quantize(x, bits=4, group_size=64, mode="nf4")
```

Result:

- Status: `ok`
- Python: `3.11.7`
- MLX version: `0.32.0.dev20260613+eccb857`
- MLX binary: `/private/tmp/nlm2pr-macos26-m2pro16gb-clean-pull-20260614T021404Z/.venv/lib/python3.11/site-packages/mlx/core.cpython-311-darwin.so`

Note: an earlier probe variant in this smoke assumed the quantize call returned multiple unpackable tensors and failed with `ValueError: not enough values to unpack`. That was a probe bug, not an NF4 support failure. The corrected no-arity-assumption probe passed.

## Hugging Face State

- Auth check: `hf auth whoami` returned `user=lyonsno orgs=mlx-community,BasinShapers`
- Cache state: `hf cache list` showed `model/ideogram-ai/ideogram-4-nf4` present, size `16.1G`
- This was therefore a warm-cache model run, not a first-download run.

## Generation Command

```sh
.venv/bin/python generate.py \
  --prompt '{"prompt":"a clean product photo of a translucent cassette labeled NF4"}' \
  --output "/Users/noahlyons/dev/mlx-ideogram4/evidence/clean_smokes/nlm2pr-macos26-m2pro16gb-clean-pull-20260614T021404Z/clean_smoke_product_nf4.png" \
  --height 512 \
  --width 512 \
  --seed 42 \
  --preset V4_DEFAULT_20 \
  --steps 20 \
  --receipt "/Users/noahlyons/dev/mlx-ideogram4/evidence/clean_smokes/nlm2pr-macos26-m2pro16gb-clean-pull-20260614T021404Z/clean_smoke_product_nf4_receipt.json"
```

Generation window:

- Start UTC: `2026-06-14T02:35:57Z`
- End UTC: `2026-06-14T02:52:09Z`

## Generation Result

- Route: `nf4-mlx-metal`
- Model: `ideogram-ai/ideogram-4-nf4`
- Quant format: `bitsandbytes NF4 (4-bit, blocksize=64)`
- Backend: `MLX + custom NF4 Metal kernels`
- Resolution: `512x512`
- Steps: `20`
- Seed: `42`
- Preset: `V4_DEFAULT_20`
- Text encoder time: `52.1s`
- Sampling time: `821.3s`
- Time per step: `41.07s`
- Total time: `971.9s`
- Peak memory: `11.51 GB`
- Active memory at receipt: `0.32 GB`
- Image tokens: `1024`
- Text tokens: `23`
- Pixel range: `[0, 255]`
- Pixel std: `51.7`

Output:

- Image: `clean_smoke_product_nf4.png`
- Image SHA256: `6b62cc18a8eb7d82d6c2ebf2057b788f8bc24c2e1b8a0d2a1955dbda627fe01f`
- JSON receipt: `clean_smoke_product_nf4_receipt.json`
- JSON receipt SHA256: `8088618609f49104a02ff7465fb4bf9e7a851d3a7d293b65b3c8a541a6307993`

Visual inspection: pass. The generated image is a clean product-style translucent cassette with readable `NF4` text on the label.

## Sharp Edges Observed

- README currently tells users to run `huggingface-cli login`; installed `huggingface_hub==1.19.0` says `huggingface-cli` is deprecated and no longer works, and points users to `hf auth login`.
- `hf cache scan` is not available in this installed HF CLI. The working cache command is `hf cache list`.
- `transformers` emitted `PyTorch was not found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.` The smoke still completed; this appears to be harmless for this MLX route.
- The initial editable install does install stock MLX first. The documented "install NF4 fork last" step is necessary and was sufficient on macOS 26.

## Supporting Logs

- `preflight.txt`
- `01_git_clone.log`
- `02_venv.log`
- `03_python_version.log`
- `04_pip_version.log`
- `05_pip_install_editable.log`
- `06_pip_install_mlx_nf4_last.log`
- `07_nf4_probe.log` (failed probe shape)
- `07b_nf4_probe_corrected.log` (passing NF4 probe)
- `08_pip_freeze.txt`
- `09_py_compile.log`
- `10_generate_help.log`
- `11_huggingface_whoami.log` (`huggingface-cli` deprecation)
- `11b_hf_auth_whoami.log`
- `12_huggingface_scan_cache.log` (`huggingface-cli` deprecation)
- `12b_hf_cache_scan.log` (`hf cache scan` not available)
- `12c_hf_cache_list.log`
- `13_generation_command.txt`
- `14_generation.log`
- `clean_smoke_product_nf4_receipt.json`
