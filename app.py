"""Gradio interface for Ideogram4 NF4 on Apple Silicon."""

import sys
import os
import json
import time
import dataclasses
import gc

sys.path.insert(0, os.path.dirname(__file__))

# Optional: override MLX and mlx-vlm paths via environment variables.
_MLX_PATH = os.environ.get("MLX_FORK_PATH", "")
if _MLX_PATH and os.path.isdir(_MLX_PATH):
    sys.path.insert(0, _MLX_PATH)
_VLM_PATH = os.environ.get("MLX_VLM_PATH", "")
if _VLM_PATH and os.path.isdir(_VLM_PATH):
    sys.path.insert(0, _VLM_PATH)

import mlx.core as mx
import numpy as np
from PIL import Image
import gradio as gr

from scheduler import LogitNormalSchedule, make_step_intervals
from transformer import Ideogram4Transformer
from load_weights import load_nf4_transformer
from load_text_encoder import load_nf4_text_encoder
from pipeline import build_inputs, LATENT_SHIFT, LATENT_SCALE
from vae import Decoder, decode_latents

ACTIVATION_LAYERS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35)

PRESETS = {
    "V4_TURBO_12": {"steps": 12, "mu": 0.5, "std": 1.75,
                     "guidance": (3.0,)*1 + (7.0,)*11},
    "V4_DEFAULT_20": {"steps": 20, "mu": 0.0, "std": 1.75,
                       "guidance": (3.0,)*2 + (7.0,)*18},
    "V4_QUALITY_48": {"steps": 48, "mu": 0.0, "std": 1.5,
                       "guidance": (3.0,)*3 + (7.0,)*45},
}

DEFAULT_MAX_RESOLUTION = 1024
PUBLIC_MAX_RESOLUTION = 512
PUBLIC_PRESET = "V4_TURBO_12"
MAX_PROMPT_CHARS = 2000
PUBLIC_MODE_REQUESTED = "--public" in sys.argv[1:]

# Global model state — load once, reuse
_state = {}

# Rate limiting
import threading
_rate_lock = threading.Lock()
_last_gen_time = 0.0
_MIN_COOLDOWN = 30.0  # seconds between generations


def _assert_nf4_available():
    """Fail loud if the active MLX install lacks NF4 support."""
    try:
        w = mx.random.normal((64, 64)).astype(mx.float16)
        q = mx.quantize(w, bits=4, group_size=64, mode="nf4")
        mx.eval(q[0])
    except Exception as e:
        raise RuntimeError(
            "\nNF4 support is NOT active in the current MLX install.\n"
            f"  (probe failed: {type(e).__name__}: {e})\n\n"
            "NF4 lives only in the fork, not PyPI MLX. Reinstall the fork LAST:\n\n"
            "    pip install --force-reinstall --no-deps "
            "git+https://github.com/lyonsno/mlx.git@nf4\n\n"
            "Or point MLX_FORK_PATH at a built fork checkout's python/ dir.\n"
        ) from e


def _coerce_int(value, default, minimum, maximum):
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def _load_models(progress=gr.Progress()):
    """Load all models once."""
    if "loaded" in _state:
        return

    _assert_nf4_available()

    token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
    model_id = "ideogram-ai/ideogram-4-nf4"

    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    from mlx_vlm.models.qwen3_vl.config import ModelConfig, TextConfig, VisionConfig
    from mlx_vlm.models.qwen3_vl.qwen3_vl import Model as Qwen3VLModel

    # Tokenizer
    progress(0.05, desc="Loading tokenizer...")
    tok_dir = os.path.dirname(hf_hub_download(model_id, "tokenizer/tokenizer.json", token=token))
    hf_hub_download(model_id, "tokenizer/chat_template.jinja", token=token)
    hf_hub_download(model_id, "tokenizer/tokenizer_config.json", token=token)
    _state["tokenizer"] = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)

    # Text encoder
    progress(0.1, desc="Loading text encoder (8.8B NF4)...")
    cfg_path = hf_hub_download(model_id, "text_encoder/config.json", token=token)
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
    text_model = Qwen3VLModel(cfg)
    wp = hf_hub_download(model_id, "text_encoder/model.safetensors", token=token)
    load_nf4_text_encoder(wp, text_model, verbose=False)
    _state["text_model"] = text_model

    # Transformers
    progress(0.4, desc="Loading conditional transformer (9.3B NF4)...")
    cf = hf_hub_download(model_id, "transformer/diffusion_pytorch_model.safetensors", token=token)
    cond = Ideogram4Transformer()
    load_nf4_transformer(cf, cond, verbose=False)
    _state["cond_model"] = cond

    progress(0.6, desc="Loading unconditional transformer (9.3B NF4)...")
    uf = hf_hub_download(model_id, "unconditional_transformer/diffusion_pytorch_model.safetensors", token=token)
    uncond = Ideogram4Transformer()
    load_nf4_transformer(uf, uncond, verbose=False)
    _state["uncond_model"] = uncond

    # VAE
    progress(0.8, desc="Loading VAE decoder...")
    vf = hf_hub_download(model_id, "vae/diffusion_pytorch_model.safetensors", token=token)
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
    _state["decoder"] = dec

    _state["token"] = token
    _state["model_id"] = model_id
    _state["loaded"] = True
    progress(1.0, desc="Ready!")


def generate(prompt_text, use_json, seed, preset_name, width, height, progress=gr.Progress()):
    """Generate an image from a text prompt."""
    global _last_gen_time

    max_resolution = PUBLIC_MAX_RESOLUTION if PUBLIC_MODE_REQUESTED else DEFAULT_MAX_RESOLUTION
    width = _coerce_int(width, 512, 256, max_resolution)
    height = _coerce_int(height, 512, 256, max_resolution)
    try:
        seed = int(seed)
    except (TypeError, ValueError):
        seed = 42

    prompt_text = str(prompt_text or "")
    if len(prompt_text) > MAX_PROMPT_CHARS:
        yield None, f"Prompt is too long ({len(prompt_text)} chars); limit is {MAX_PROMPT_CHARS}."
        return

    # Rate limit only accepted generation requests.
    with _rate_lock:
        now = time.time()
        elapsed = now - _last_gen_time
        if elapsed < _MIN_COOLDOWN and _last_gen_time > 0:
            wait = int(_MIN_COOLDOWN - elapsed)
            yield None, f"Rate limited — please wait {wait}s before generating again"
            return
        _last_gen_time = now

    _load_models(progress)

    # In JSON mode, pass through raw. Otherwise wrap plain text.
    if use_json:
        prompt = prompt_text
    else:
        prompt = json.dumps({"prompt": prompt_text})

    if PUBLIC_MODE_REQUESTED:
        preset_name = PUBLIC_PRESET
    elif preset_name not in PRESETS:
        preset_name = "V4_DEFAULT_20"
    preset = PRESETS[preset_name]
    num_steps = preset["steps"]
    guidance = preset["guidance"]

    # Yield loading state immediately
    yield None, f"Loading models... ({num_steps} steps, {int(width)}×{int(height)})"

    import math
    schedule_mean = preset["mu"]
    if width != 512 or height != 512:
        schedule_mean += 0.5 * math.log(width * height / (512 * 512))
    schedule = LogitNormalSchedule(mean=schedule_mean, std=preset["std"])
    steps = make_step_intervals(num_steps)

    # Tokenize
    yield None, "Tokenizing prompt..."
    tokenizer = _state["tokenizer"]
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    token_ids_np = tokenizer(text, return_tensors="np", add_special_tokens=False)["input_ids"][0]
    num_text_tokens = len(token_ids_np)

    # Text encoder
    yield None, f"Encoding text ({num_text_tokens} tokens)..."
    tm = _state["text_model"]
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

    # Build inputs
    inp = build_inputs(num_text_tokens, height, width)
    ni = inp["num_image_tokens"]
    tl = num_text_tokens + ni
    lf = mx.zeros((1, tl, 53248), dtype=mx.bfloat16)
    lf = lf.at[:, :num_text_tokens, :].add(lt)
    mx.eval(lf)

    neg_pos = inp["position_ids"][:, num_text_tokens:]
    neg_seg = inp["segment_ids"][:, num_text_tokens:]
    neg_ind = inp["indicator"][:, num_text_tokens:]
    neg_llm = mx.zeros((1, ni, 53248), dtype=mx.bfloat16)
    tp = mx.zeros((1, num_text_tokens, 128))

    # Sample
    yield None, f"Starting diffusion ({ni} image tokens, {num_steps} steps)..."
    seed = int(seed)
    mx.random.seed(seed)
    z = mx.random.normal((1, ni, 128))

    cond_model = _state["cond_model"]
    uncond_model = _state["uncond_model"]

    # Preview every ~5 steps locally; skip in public mode to protect memory.
    previews_enabled = (
        not PUBLIC_MODE_REQUESTED and os.environ.get("NF4_NO_PREVIEW") != "1"
    )
    preview_interval = max(1, min(5, num_steps // 4))
    decoder = _state["decoder"]
    gh, gw_grid = inp["grid_h"], inp["grid_w"]

    # Reset peak memory to capture sampling-only peak
    mx.get_peak_memory()  # read to clear
    try:
        mx.reset_peak_memory()
    except AttributeError:
        pass  # older MLX versions

    t0 = time.perf_counter()
    for i in range(num_steps - 1, -1, -1):
        step_num = num_steps - i
        progress(step_num / num_steps * 0.8 + 0.1,
                 desc=f"Sampling step {step_num}/{num_steps}...")
        tv = schedule(steps[i + 1:i + 2]).item()
        sv = schedule(steps[i:i + 1]).item()
        t = mx.array([tv])
        gw = guidance[i]
        pz = mx.concatenate([tp, z], axis=1)
        pv = cond_model(llm_features=lf, x=pz.astype(mx.bfloat16), t=t,
                        position_ids=inp["position_ids"],
                        segment_ids=inp["segment_ids"],
                        indicator=inp["indicator"])[:, num_text_tokens:]
        nv = uncond_model(llm_features=neg_llm, x=z.astype(mx.bfloat16), t=t,
                          position_ids=neg_pos, segment_ids=neg_seg,
                          indicator=neg_ind)
        v = gw * pv + (1.0 - gw) * nv
        z = z + v * (sv - tv)
        mx.eval(z)

        # Yield preview at intervals — decode and display intermediate result
        if step_num % preview_interval == 0 and step_num < num_steps:
            elapsed = time.perf_counter() - t0
            if previews_enabled:
                try:
                    preview_pixels = decode_latents(decoder, z, gh, gw_grid, LATENT_SHIFT, LATENT_SCALE)
                    mx.eval(preview_pixels)
                    preview_np = np.array(preview_pixels[0]).transpose(1, 2, 0)
                    preview_img = Image.fromarray(preview_np)
                except Exception:
                    preview_img = None
            else:
                preview_img = None
            step_info = (f"Step {step_num}/{num_steps} | {elapsed:.0f}s elapsed | "
                         f"{elapsed/step_num:.1f}s/step")
            yield preview_img, step_info

    sampling_time = time.perf_counter() - t0
    sampling_peak = mx.get_peak_memory() / 1e9

    # Final VAE decode
    progress(0.95, desc="Final decode...")
    pixels = decode_latents(decoder, z, gh, gw_grid, LATENT_SHIFT, LATENT_SCALE)
    mx.eval(pixels)
    total_peak = mx.get_peak_memory() / 1e9

    pn = np.array(pixels[0]).transpose(1, 2, 0)
    img = Image.fromarray(pn)

    active = mx.get_active_memory() / 1e9
    info = (f"{int(width)}×{int(height)} | {num_steps} steps | {sampling_time:.0f}s sampling "
            f"({sampling_time/num_steps:.1f}s/step) | seed {int(seed)}\n"
            f"Memory: {active:.1f} GB active | {sampling_peak:.1f} GB sampling peak | "
            f"{total_peak:.1f} GB total peak")

    yield img, info


# Build UI
_preset_choices = [PUBLIC_PRESET] if PUBLIC_MODE_REQUESTED else list(PRESETS.keys())
_default_preset = PUBLIC_PRESET if PUBLIC_MODE_REQUESTED else "V4_DEFAULT_20"
_slider_max = PUBLIC_MAX_RESOLUTION if PUBLIC_MODE_REQUESTED else DEFAULT_MAX_RESOLUTION

with gr.Blocks(title="Ideogram4 NF4 — Apple Silicon") as demo:
    gr.Markdown("""
    # Ideogram4 NF4 on Apple Silicon
    *9.3B parameter text-to-image through custom NF4 Metal kernels. 11.5 GB peak memory.*
    """)

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder='a red cat sitting on a blue couch',
                lines=2,
                value='a red cat sitting on a blue couch',
            )
            use_json = gr.Checkbox(label="Advanced JSON mode", value=False,
                                   info="Edit raw JSON prompt (for style/layout control)")
            with gr.Row():
                seed = gr.Number(label="Seed", value=42, precision=0)
                preset = gr.Dropdown(
                    label="Preset",
                    choices=_preset_choices,
                    value=_default_preset,
                )
            with gr.Row():
                width = gr.Slider(256, _slider_max, value=512, step=16, label="Width")
                height = gr.Slider(256, _slider_max, value=512, step=16, label="Height")
            btn = gr.Button("Generate", variant="primary", size="lg")

        with gr.Column(scale=1):
            output_image = gr.Image(label="Diffusion Preview", type="pil",
                                    height=512)
            info = gr.Textbox(label="Status", interactive=False,
                              value="Ready — select a prompt and click Generate")

    gr.Examples(
        examples=[
            ["a red cat sitting on a blue couch"],
            ["the word HELLO written in neon lights on a brick wall at night"],
            ["a cup of coffee with latte art on a wooden table, morning light"],
            ["bold black letters NF4 inside an Apple logo silhouette, minimal graphic design, white background"],
            ["a vintage travel poster for Mars, retro 1960s NASA screenprint style, text reads VISIT MARS"],
            ["a cozy bookshop interior, golden hour light through windows, cat curled up in an armchair"],
        ],
        inputs=[prompt],
        label="Examples (click to use)",
    )

    btn.click(fn=generate,
              inputs=[prompt, use_json, seed, preset, width, height],
              outputs=[output_image, info],
              show_progress="minimal")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create public Gradio URL")
    parser.add_argument("--auth", type=str, default=None, help="user:password for basic auth")
    parser.add_argument("--public", action="store_true",
                        help="Public mode: strict queue, rate limits, 512 max, turbo only")
    args = parser.parse_args()

    if args.public:
        # Lock down for public hosting:
        # - Queue size 1 (one person waits, everyone else gets "queue full")
        # - Server-side 512 max resolution, turbo preset, prompt cap, no previews
        demo.queue(max_size=1, default_concurrency_limit=1)
        print("PUBLIC MODE: queue=1, one job at a time", flush=True)
    else:
        demo.queue(max_size=2)

    auth = None
    if args.auth:
        user, pw = args.auth.split(":", 1)
        auth = (user, pw)

    demo.launch(share=args.share, auth=auth)
