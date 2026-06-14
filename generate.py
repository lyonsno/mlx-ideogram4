#!/usr/bin/env python3
"""Ideogram4 NF4 image generation on Apple Silicon.

One-command generation with route identity receipt.

Usage:
    python generate.py --prompt '{"prompt": "a red cat on a blue couch"}' --output cat.png
    python generate.py --prompt '{"prompt": "bold text NF4"}' --seed 2025 --steps 20
"""

import argparse
import json
import os
import sys
import time
import dataclasses
import gc
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "."))

# Optional: override MLX and mlx-vlm paths via environment variables.
# Only needed for development — pip install handles this normally.
_MLX_PATH = os.environ.get("MLX_FORK_PATH", "")
if _MLX_PATH and os.path.isdir(_MLX_PATH):
    sys.path.insert(0, _MLX_PATH)

_VLM_PATH = os.environ.get("MLX_VLM_PATH", "")
if _VLM_PATH and os.path.isdir(_VLM_PATH):
    sys.path.insert(0, _VLM_PATH)

import mlx.core as mx
import numpy as np
from PIL import Image


def _assert_nf4_available():
    """Fail loud if the active MLX lacks NF4 support.

    Stock PyPI MLX does not implement quantization mode 'nf4' — it lives only
    in the lyonsno/mlx@nf4 fork. A mixed install (fork's C++ core but stock
    Python layers, or stock MLX silently reinstalled by an mlx-vlm dependency)
    otherwise surfaces as a cryptic `KeyError: 'nf4'` deep inside a layer
    constructor mid-run, after the model has started loading. Probe up front
    so the failure names its own fix.
    """
    try:
        w = mx.random.normal((64, 64)).astype(mx.float16)
        q = mx.quantize(w, bits=4, group_size=64, mode="nf4")
        mx.eval(q[0])
    except Exception as e:
        raise SystemExit(
            "\nNF4 support is NOT active in the current MLX install.\n"
            f"  (probe failed: {type(e).__name__}: {e})\n\n"
            "NF4 lives only in the fork, not PyPI MLX. Most likely a stock MLX\n"
            "got installed (often pulled in transitively by mlx-lm/mlx-vlm) and\n"
            "shadowed the fork. Reinstall the fork LAST:\n\n"
            "    pip install --force-reinstall --no-deps "
            "git+https://github.com/lyonsno/mlx.git@nf4\n\n"
            "Verify with:  python -c \"import mlx.core as mx; "
            "mx.quantize(mx.zeros((64,64)).astype(mx.float16), "
            "bits=4, group_size=64, mode='nf4')\"\n"
            "Or point MLX_FORK_PATH at a built fork checkout's python/ dir.\n"
        )


def _get_hf_token():
    from huggingface_hub.utils import get_token

    token = get_token()
    if not token:
        raise RuntimeError(
            "Hugging Face token not found. Run `hf auth login` or set HF_TOKEN "
            "after accepting the Ideogram 4 NF4 license."
        )
    return token


ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)

PRESETS = {
    "V4_QUALITY_48": {"steps": 48, "mu": 0.0, "std": 1.5,
                       "guidance": (3.0,)*3 + (7.0,)*45},
    "V4_DEFAULT_20": {"steps": 20, "mu": 0.0, "std": 1.75,
                       "guidance": (3.0,)*2 + (7.0,)*18},
    "V4_TURBO_12":   {"steps": 12, "mu": 0.5, "std": 1.75,
                       "guidance": (3.0,)*1 + (7.0,)*11},
}


def main():
    parser = argparse.ArgumentParser(description="Ideogram4 NF4 on Apple Silicon")
    parser.add_argument("--prompt", required=True, help="JSON prompt string")
    parser.add_argument("--output", default="output.png")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preset", default="V4_DEFAULT_20", choices=PRESETS.keys())
    parser.add_argument("--steps", type=int, default=None, help="Override preset steps")
    parser.add_argument("--model", default="ideogram-ai/ideogram-4-nf4")
    parser.add_argument("--receipt", default=None, help="Write JSON receipt to this path")
    args = parser.parse_args()

    # Probe NF4 support before any model work, but after argparse so --help
    # still works on a broken install. Fails loud with the exact fix.
    _assert_nf4_available()

    preset = PRESETS[args.preset]
    num_steps = args.steps or preset["steps"]
    guidance = preset["guidance"]
    if len(guidance) != num_steps:
        # Pad/trim guidance to match steps
        if num_steps > len(guidance):
            guidance = (7.0,) * (num_steps - 3) + (3.0,) * 3
        else:
            guidance = guidance[:num_steps]

    receipt = {
        "route": "nf4-mlx-metal",
        "model": args.model,
        "quant_format": "bitsandbytes NF4 (4-bit, blocksize=64)",
        "backend": "MLX + custom NF4 Metal kernels",
        "prompt": args.prompt,
        "seed": args.seed,
        "resolution": f"{args.width}x{args.height}",
        "steps": num_steps,
        "preset": args.preset,
        "device": "Apple Silicon",
        "output": os.path.abspath(args.output),
    }

    t_total = time.perf_counter()

    # === Imports ===
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download
    from scheduler import LogitNormalSchedule, make_step_intervals
    import math

    token = _get_hf_token()

    schedule_mean = preset["mu"] + 0.5 * math.log(
        args.height * args.width / (512 * 512)
    ) if args.height != 512 or args.width != 512 else preset["mu"]
    schedule = LogitNormalSchedule(mean=schedule_mean, std=preset["std"])
    steps = make_step_intervals(num_steps)

    # === Tokenize ===
    print(f"Ideogram4 NF4 | {args.width}x{args.height} | {num_steps} steps | seed {args.seed}", flush=True)
    tok_dir = os.path.dirname(hf_hub_download(args.model, "tokenizer/tokenizer.json", token=token))
    hf_hub_download(args.model, "tokenizer/chat_template.jinja", token=token)
    hf_hub_download(args.model, "tokenizer/tokenizer_config.json", token=token)
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)

    messages = [{"role": "user", "content": [{"type": "text", "text": args.prompt}]}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    token_ids_np = tokenizer(text, return_tensors="np", add_special_tokens=False)["input_ids"][0]
    num_text_tokens = len(token_ids_np)

    # === Text encoder ===
    print(f"Text encoder ({num_text_tokens} tokens)...", flush=True)
    from mlx_vlm.models.qwen3_vl.config import ModelConfig, TextConfig, VisionConfig
    from mlx_vlm.models.qwen3_vl.qwen3_vl import Model as Qwen3VLModel
    from load_text_encoder import load_nf4_text_encoder

    cfg_path = hf_hub_download(args.model, "text_encoder/config.json", token=token)
    with open(cfg_path) as fp:
        raw = json.load(fp)
    raw.pop("quantization_config", None)
    tr = dict(raw["text_config"])
    rp = tr.pop("rope_parameters", {})
    tr["rope_scaling"] = {"type": rp.get("rope_type", "default"),
                          "mrope_section": rp.get("mrope_section", [24, 20, 20])}
    tr.setdefault("rope_theta", rp.get("rope_theta", 5000000))
    tcf = {f.name for f in dataclasses.fields(TextConfig)}
    vcf = {f.name for f in dataclasses.fields(VisionConfig)}
    tc = TextConfig(**{k: v for k, v in tr.items() if k in tcf})
    vr = dict(raw["vision_config"])
    vr["model_type"] = "qwen3_vl"
    vc = VisionConfig(**{k: v for k, v in vr.items() if k in vcf})
    cfg = ModelConfig(text_config=tc, vision_config=vc, model_type="qwen3_vl",
                      image_token_id=raw.get("image_token_id", 151655))
    tm = Qwen3VLModel(cfg)
    wp = glob.glob(os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{args.model.replace('/', '--')}/snapshots/*/text_encoder/model.safetensors"
    ))[0]
    load_nf4_text_encoder(wp, tm, verbose=False)

    ids = mx.array(token_ids_np[None, :])
    B, L = ids.shape
    pos = mx.broadcast_to(mx.arange(L).reshape(1, 1, -1), (3, B, L))
    ln = tm.language_model.model
    h = ln.embed_tokens(ids)
    cm = mx.where(mx.tril(mx.ones((L, L))),
                  mx.array(0.0, dtype=mx.bfloat16),
                  mx.array(-1e9, dtype=mx.bfloat16))[None, None]
    cap = {}
    for i, layer in enumerate(ln.layers):
        h = layer(h, cm, None, pos)
        if i in set(ACTIVATION_LAYERS):
            cap[i] = h
        if i % 9 == 0:
            mx.eval(h)
    mx.eval(h)
    fs = [cap[i] for i in ACTIVATION_LAYERS]
    st = mx.transpose(mx.stack(fs, axis=0), (1, 2, 3, 0))
    lt = mx.reshape(st, (B, L, -1)).astype(mx.bfloat16)
    mx.eval(lt)
    del tm, ln, cap, fs, st, h
    gc.collect()
    t_text = time.perf_counter() - t_total

    # === Sampling ===
    print("Loading transformers...", flush=True)
    from transformer import Ideogram4Transformer
    from load_weights import load_nf4_transformer
    from pipeline import build_inputs, LATENT_SHIFT, LATENT_SCALE

    cf = hf_hub_download(args.model, "transformer/diffusion_pytorch_model.safetensors", token=token)
    cm_ = Ideogram4Transformer()
    load_nf4_transformer(cf, cm_, verbose=False)
    uf = hf_hub_download(args.model, "unconditional_transformer/diffusion_pytorch_model.safetensors", token=token)
    um = Ideogram4Transformer()
    load_nf4_transformer(uf, um, verbose=False)

    inp = build_inputs(num_text_tokens, args.height, args.width)
    ni = inp["num_image_tokens"]
    tl = num_text_tokens + ni
    lf = mx.zeros((1, tl, 53248), dtype=mx.bfloat16)
    lf = lf.at[:, :num_text_tokens, :].add(lt)
    mx.eval(lf)
    del lt
    gc.collect()

    np_ = inp["position_ids"][:, num_text_tokens:]
    ns = inp["segment_ids"][:, num_text_tokens:]
    nind = inp["indicator"][:, num_text_tokens:]
    nl = mx.zeros((1, ni, 53248), dtype=mx.bfloat16)
    tp = mx.zeros((1, num_text_tokens, 128))

    mx.random.seed(args.seed)
    z = mx.random.normal((1, ni, 128))

    print(f"Sampling {num_steps} steps ({ni} tokens)...", flush=True)
    t0 = time.perf_counter()
    for i in range(num_steps - 1, -1, -1):
        tv = schedule(steps[i + 1:i + 2]).item()
        sv = schedule(steps[i:i + 1]).item()
        t = mx.array([tv])
        gw = guidance[i]
        pz = mx.concatenate([tp, z], axis=1)
        pv = cm_(llm_features=lf, x=pz.astype(mx.bfloat16), t=t,
                 position_ids=inp["position_ids"], segment_ids=inp["segment_ids"],
                 indicator=inp["indicator"])[:, num_text_tokens:]
        nv = um(llm_features=nl, x=z.astype(mx.bfloat16), t=t,
                position_ids=np_, segment_ids=ns, indicator=nind)
        v = gw * pv + (1.0 - gw) * nv
        z = z + v * (sv - tv)
        mx.eval(z)
        if i % max(1, num_steps // 4) == 0 or i == num_steps - 1:
            print(f"  {num_steps - i}/{num_steps}", flush=True)

    t_sampling = time.perf_counter() - t0
    del cm_, um, lf
    gc.collect()

    # === VAE decode ===
    print("VAE decode...", flush=True)
    from vae import Decoder, decode_latents

    vf = hf_hub_download(args.model, "vae/diffusion_pytorch_model.safetensors", token=token)
    vw = mx.load(vf)
    dec = Decoder()
    mp = []
    for k, v in vw.items():
        if k.startswith("decoder."):
            nk = k[len("decoder."):]
        elif k in ("post_quant_conv.weight", "post_quant_conv.bias"):
            nk = k
        else:
            continue
        nk = nk.replace("mid_block.resnets.0.", "mid_block_1.")
        nk = nk.replace("mid_block.resnets.1.", "mid_block_2.")
        nk = nk.replace("mid_block.attentions.0.", "mid_attn_1.")
        nk = nk.replace(".upsamplers.0.", ".upsamplers.0.")
        nk = nk.replace("conv_norm_out.", "norm_out.")
        nk = nk.replace("to_q.", "q.").replace("to_k.", "k.").replace("to_v.", "v.")
        nk = nk.replace("to_out.0.", "proj_out.").replace("group_norm.", "norm.")
        nk = nk.replace("conv_shortcut.", "nin_shortcut.")
        if "weight" in k and v.ndim == 4:
            v = mx.transpose(v, (0, 2, 3, 1))
        if "norm" in nk and v.ndim == 1:
            v = v.reshape(1, 1, 1, -1)
        mp.append((nk, v))
    dec.load_weights(mp, strict=False)

    gh, gw = inp["grid_h"], inp["grid_w"]
    pixels = decode_latents(dec, z, gh, gw, LATENT_SHIFT, LATENT_SCALE)
    mx.eval(pixels)

    pn = np.array(pixels[0]).transpose(1, 2, 0)
    img = Image.fromarray(pn)
    img.save(args.output)

    t_total_elapsed = time.perf_counter() - t_total

    # === Receipt ===
    receipt.update({
        "time_text_encoder_s": round(t_text, 1),
        "time_sampling_s": round(t_sampling, 1),
        "time_per_step_s": round(t_sampling / num_steps, 2),
        "time_total_s": round(t_total_elapsed, 1),
        "memory_active_gb": round(mx.get_active_memory() / 1e9, 2),
        "memory_peak_gb": round(mx.get_peak_memory() / 1e9, 2),
        "image_tokens": ni,
        "text_tokens": num_text_tokens,
        "pixel_range": [int(pn.min()), int(pn.max())],
        "pixel_std": round(float(pn.std()), 1),
    })

    print(f"\nSaved: {args.output} ({img.size[0]}x{img.size[1]})", flush=True)
    print(f"Sampling: {t_sampling:.0f}s ({t_sampling/num_steps:.1f}s/step)", flush=True)
    print(f"Total: {t_total_elapsed:.0f}s", flush=True)
    print(f"Memory: {receipt['memory_peak_gb']} GB peak", flush=True)

    receipt_path = args.receipt or args.output.replace(".png", "_receipt.json")
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2)
    print(f"Receipt: {receipt_path}", flush=True)


if __name__ == "__main__":
    main()
