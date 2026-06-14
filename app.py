"""Gradio interface for Ideogram4 NF4 on Apple Silicon."""

import sys
import os
import json
import time
import dataclasses
import gc
import hashlib
import glob
import subprocess
import selectors
import random
import html
import urllib.parse

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

MODEL_ID = "ideogram-ai/ideogram-4-nf4"
MODEL_REVISION = "f664347839e0a87bc495f5c9483cc0014b8e344e"
ROUTE_ID = "nf4-mlx-metal"
QUANT_FORMAT = "bitsandbytes NF4 (4-bit, blocksize=64)"
BACKEND = "MLX + custom NF4 Metal kernels"

# Global model state — load once, reuse
_state = {}

# Rate limiting
import threading
_rate_lock = threading.Lock()
_last_gen_time = 0.0
_MIN_COOLDOWN = 30.0  # seconds between generations

PUBLIC_WIDTH = 512
PUBLIC_HEIGHT = 512
PUBLIC_MIN_WIDTH = 256
PUBLIC_MIN_HEIGHT = 256
PUBLIC_MAX_WIDTH = 512
PUBLIC_MAX_HEIGHT = 512
PUBLIC_STEPS = 20
PUBLIC_MIN_STEPS = 4
PUBLIC_MAX_STEPS = 20
PUBLIC_PRESET = "V4_DEFAULT_20"
PUBLIC_EXPECTED_LATENCY = "10-15 minutes"
PUBLIC_HARDWARE = "16 GB M2 Pro MacBook Pro"
PUBLIC_QUEUE_SIZE = 10
PUBLIC_STATUS_HEARTBEAT_SECONDS = 5
QUEUE_STATUS_TIMER_SECONDS = 3
QUEUE_STATUS_ELEM_ID = "public-queue-status"
QUEUE_ADMIN_TIMER_SECONDS = 3
QUEUE_ADMIN_ELEM_ID = "admin-queue-status"
PUBLIC_GALLERY_FILES = [
    "evidence/smoke_16gb_512_20step.png",
    "evidence/hero_nf4_apple_1024x1024.png",
]
GALLERY_ROTATE_SECONDS = 5
GALLERY_SLOT_COUNT = 40
GALLERY_CONSOLE_ELEM_ID = "gallery-visibility-console"
GALLERY_ADMIN_QUERY_PARAM = "gallery_admin"
GALLERY_ADMIN_QUERY_VALUE = "1"
GALLERY_MANIFEST_PATH = os.path.join(os.path.dirname(__file__), "evidence", "gallery_manifest.json")
REQUEST_LOG_PATH = os.path.join(os.path.dirname(__file__), "evidence", "live_runs", "requests.jsonl")
PRESET_PLACEHOLDER = "Pick a preset"
_queue_lock = threading.Lock()
_queue_active = 0
_queue_waiting = 0
_queue_next_id = 0
_queue_tokens = {}
_queue_records = {}

PROMPT_PRESETS = [
    ("Red cat on couch", "a red cat sitting on a blue couch"),
    ("Text poster", "the word HELLO written in neon lights on a brick wall at night"),
    ("NF4 Apple mark", "bold black letters NF4 inside an Apple logo silhouette"),
    ("Coffee study", "a cup of coffee with latte art on a wooden table"),
    ("Bookshop cat", "a cozy bookshop interior with a cat curled up in an armchair"),
    ("Product mockup", "a clean product photo of a translucent cassette labeled NF4"),
    ("Sticker mascot", "a cheerful robot sticker holding a tiny metal kernel"),
    ("Mountain lake", "a serene mountain lake at sunset with snow-capped peaks reflected in still water"),
]
AESTHETICS_PRESETS = [
    ("Minimal graphic design", "minimal graphic design"),
    ("Clean product photo", "clean product photography"),
    ("Editorial magazine", "editorial magazine spread"),
    ("Playful sticker", "playful vinyl sticker style"),
    ("Cinematic realism", "cinematic realism"),
    ("Retro computer ad", "1980s computer magazine advertisement"),
    ("Soft painterly", "soft painterly illustration"),
    ("Bold typography", "bold typography-focused poster design"),
]
LIGHTING_PRESETS = [
    ("Flat studio", "flat studio lighting"),
    ("Golden hour", "warm golden hour light"),
    ("Neon night", "neon signage at night"),
    ("Soft window", "soft window light"),
    ("High-key", "bright high-key lighting"),
    ("Low-key", "moody low-key lighting"),
    ("Backlit rim", "backlit with crisp rim light"),
    ("Overcast", "diffuse overcast daylight"),
]
MEDIUM_PRESETS = [
    ("Digital vector", "digital vector art"),
    ("Photograph", "high-resolution photograph"),
    ("Screen print", "two-color screen print"),
    ("Oil painting", "oil painting on canvas"),
    ("3D render", "polished 3D render"),
    ("Risograph", "risograph print"),
    ("Ink drawing", "clean ink drawing"),
    ("UI mockup", "product interface mockup"),
]
COLOR_PALETTE_PRESETS = [
    ("Black / white", "#000000, #FFFFFF"),
    ("Blue couch / orange cat", "#1F5E7A, #D76B2A, #F4D6A0"),
    ("NF4 orange", "#FF6B1A, #202020, #F7F7F2"),
    ("Neon", "#FF2D95, #00E5FF, #111111"),
    ("Forest", "#12372A, #6B8E23, #F3E9D2"),
    ("Warm paper", "#2F2A24, #E7D4B5, #B85C38"),
    ("Candy", "#FF7AB6, #7AE7FF, #FFF7A8"),
    ("Metal", "#C9CED6, #3B4452, #111820"),
]
CANVAS_PRESETS = [
    ("Square 512", "512 x 512 square canvas"),
    ("Square 384", "384 x 384 square canvas"),
    ("Small square", "256 x 256 square canvas"),
    ("Poster crop", "portrait poster crop"),
    ("Wide banner", "wide banner composition"),
    ("Album cover", "centered square album-cover crop"),
    ("Icon tile", "single app-icon tile"),
    ("Product card", "clean product-card canvas"),
]
BACKGROUND_PRESETS = [
    ("Pure white", "pure white background"),
    ("Blue couch", "deep blue couch in the background"),
    ("Brick wall", "dark brick wall background"),
    ("Warm desk", "warm wooden desk surface"),
    ("Mountain dusk", "distant mountain landscape at dusk"),
    ("Transparent feel", "plain light background with transparent-object feel"),
    ("Grid paper", "subtle graph-paper background"),
    ("Black stage", "matte black studio stage"),
]
LAYOUT_PRESETS = [
    ("Centered", "centered single subject"),
    ("Rule of thirds", "subject placed on the left third with open space"),
    ("Text dominant", "large readable text dominates the center"),
    ("Hero product", "hero object centered with small supporting details"),
    ("Diagonal", "diagonal movement from lower left to upper right"),
    ("Badge", "compact badge-like composition"),
    ("Split stack", "stacked text above object"),
    ("Symmetric", "symmetrical composition with balanced margins"),
]
ELEMENTS_PRESETS = [
    ("NF4 logo text", "text | Large bold black letters NF4\nobj | Apple logo silhouette behind the letters"),
    ("Cat portrait", "animal | Orange cat with visible whiskers\nobj | Blue couch cushions"),
    ("Coffee table", "obj | Ceramic latte cup\nobj | Wooden table\nlight | Morning light across the surface"),
    ("Neon wall", "text | The word HELLO in bright neon tubing\nbg | Dark brick wall"),
    ("Product label", "obj | Translucent cassette shell\ntext | Small label reading NF4"),
    ("Bookshop", "animal | Cat curled in a worn armchair\nobj | Tall shelves of books"),
    ("Sticker", "character | Cheerful robot mascot\nobj | Tiny glowing metal kernel"),
    ("Landscape", "env | Still mountain lake\nenv | Snow-capped peaks reflected in water"),
]
ADVANCED_RUN_PRESETS = [
    {"key": "balanced_512", "label": "512 balanced", "seed": 42, "width": 512, "height": 512, "steps": 20},
    {"key": "quick_384", "label": "384 quicker", "seed": 101, "width": 384, "height": 384, "steps": 12},
    {"key": "probe_256", "label": "256 probe", "seed": 2025, "width": 256, "height": 256, "steps": 8},
    {"key": "square_320", "label": "320 square", "seed": 31415, "width": 320, "height": 320, "steps": 12},
    {"key": "small_detail", "label": "448 detail", "seed": 777, "width": 448, "height": 448, "steps": 20},
    {"key": "wide_local", "label": "Wide local", "seed": 1234, "width": 512, "height": 320, "steps": 16},
    {"key": "tall_local", "label": "Tall local", "seed": 5678, "width": 320, "height": 512, "steps": 16},
    {"key": "text_test", "label": "Text test", "seed": 9001, "width": 512, "height": 512, "steps": 20},
]


def _is_public_mode():
    return os.environ.get("NF4_PUBLIC_MODE") == "1" or "--public" in sys.argv


def _gallery_console_visibility_css(public_mode):
    css_parts = []
    if public_mode:
        css_parts.append(f"""
#{GALLERY_CONSOLE_ELEM_ID} {{
    display: none !important;
}}
.gallery-admin-enabled #{GALLERY_CONSOLE_ELEM_ID} {{
    display: block !important;
}}
html.gallery-admin-enabled .gradio-container .contain #{GALLERY_CONSOLE_ELEM_ID},
body.gallery-admin-enabled .gradio-container .contain #{GALLERY_CONSOLE_ELEM_ID} {{
    display: block !important;
}}
""")
    css_parts.append("""
.admin-queue-panel {
    background: #18181c;
    border: 1px solid #3a3a42;
    border-radius: 8px;
    color: #f2f2f4;
    margin: 8px 0 16px;
    padding: 12px;
}
.admin-queue-header {
    align-items: center;
    display: flex;
    gap: 12px;
    justify-content: space-between;
    margin-bottom: 10px;
}
.admin-queue-header span,
.admin-queue-note {
    color: #b9bbc6;
    font-size: 13px;
}
.admin-queue-table {
    border-collapse: collapse;
    font-size: 13px;
    width: 100%;
}
.admin-queue-table th,
.admin-queue-table td {
    border-top: 1px solid #34343b;
    padding: 8px 6px;
    text-align: left;
    vertical-align: top;
}
.admin-queue-table th {
    color: #d8d9df;
    font-weight: 600;
}
.admin-queue-table code {
    color: #f2f2f4;
    white-space: nowrap;
}
.queue-state {
    border-radius: 999px;
    display: inline-block;
    font-size: 12px;
    font-weight: 700;
    line-height: 1;
    padding: 5px 8px;
}
.queue-state-active {
    background: #1d7f4f;
    color: #fff;
}
.queue-state-waiting {
    background: #6246ea;
    color: #fff;
}
.queue-empty {
    color: #b9bbc6;
    text-align: center !important;
}
""")
    return "\n".join(css_parts)


def _gallery_console_visibility_head():
    return f"""
<script>
(function() {{
    function enableGalleryAdmin() {{
        document.documentElement.classList.add("gallery-admin-enabled");
        if (document.body) {{
            document.body.classList.add("gallery-admin-enabled");
        }}
    }}
    const params = new URLSearchParams(window.location.search);
    if (params.get("{GALLERY_ADMIN_QUERY_PARAM}") === "{GALLERY_ADMIN_QUERY_VALUE}") {{
        enableGalleryAdmin();
        document.addEventListener("DOMContentLoaded", enableGalleryAdmin);
    }}
}})();
</script>
"""


def _clamp_public_dimension(value, minimum, maximum):
    value = int(value)
    value = max(minimum, min(maximum, value))
    return value - (value % 16)


def _clamp_public_steps(value):
    value = int(value)
    return max(PUBLIC_MIN_STEPS, min(PUBLIC_MAX_STEPS, value))


def _effective_request(prompt_text, use_json, preset_name, width, height, steps):
    prompt_text = (prompt_text or "").strip()
    if _is_public_mode():
        return (
            prompt_text,
            False,
            PUBLIC_PRESET,
            _clamp_public_dimension(width, PUBLIC_MIN_WIDTH, PUBLIC_MAX_WIDTH),
            _clamp_public_dimension(height, PUBLIC_MIN_HEIGHT, PUBLIC_MAX_HEIGHT),
            _clamp_public_steps(steps),
        )
    return prompt_text, bool(use_json), preset_name, min(int(width), 1024), min(int(height), 1024), int(steps)


def _public_queue_snapshot():
    with _queue_lock:
        active = int(_queue_active)
        waiting = int(_queue_waiting)
        entries = [dict(record) for record in _queue_records.values()]
    admitted = active + waiting
    free = max(0, PUBLIC_QUEUE_SIZE - admitted)
    entries.sort(key=lambda entry: (0 if entry.get("state") == "active" else 1, entry.get("sequence", 0)))
    return {
        "active": active,
        "waiting": waiting,
        "admitted": admitted,
        "free": free,
        "full": admitted >= PUBLIC_QUEUE_SIZE,
        "capacity": PUBLIC_QUEUE_SIZE,
        "entries": entries,
    }


def _public_queue_is_full():
    return _public_queue_snapshot()["full"]


def _public_queue_status_html():
    snapshot = _public_queue_snapshot()
    accent = "#b42318" if snapshot["full"] else "#1d7f4f"
    label = "Queue full" if snapshot["full"] else "Queue open"
    note = (
        "Try again in a few minutes."
        if snapshot["full"]
        else "Generate stays available until all public slots are admitted."
    )
    return (
        f"<div id='{QUEUE_STATUS_ELEM_ID}' style='border:1px solid #d8d8d8;border-left:4px solid {accent};"
        "border-radius:6px;padding:10px 12px;margin:8px 0 12px;background:#fff;font-size:14px;line-height:1.4;'>"
        f"<b>{label}</b> · Queue: {snapshot['active']} running / {snapshot['waiting']} waiting / "
        f"{snapshot['free']} free of {snapshot['capacity']} slots<br>"
        f"<span style='color:#666;'>{note}</span>"
        "</div>"
    )


def _queue_age_label(timestamp, now=None):
    if not timestamp:
        return "-"
    now = time.time() if now is None else now
    seconds = max(0, int(now - float(timestamp)))
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {seconds:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _public_queue_admin_html():
    snapshot = _public_queue_snapshot()
    now = time.time()
    rows = []
    for entry in snapshot["entries"]:
        state = html.escape(str(entry.get("state", "unknown")))
        token = html.escape(str(entry.get("token", "")))
        admitted = html.escape(_queue_age_label(entry.get("admitted_at"), now))
        started = html.escape(_queue_age_label(entry.get("started_at"), now))
        position = html.escape(str(entry.get("position", "-")))
        rows.append(
            "<tr>"
            f"<td><span class='queue-state queue-state-{state}'>{state}</span></td>"
            f"<td><code>{token}</code></td>"
            f"<td>{position}</td>"
            f"<td>{admitted}</td>"
            f"<td>{started}</td>"
            "</tr>"
        )
    if not rows:
        rows.append("<tr><td colspan='5' class='queue-empty'>No admitted public jobs right now.</td></tr>")
    return (
        f"<div id='{QUEUE_ADMIN_ELEM_ID}' class='admin-queue-panel'>"
        "<div class='admin-queue-header'>"
        "<b>Admin queue</b>"
        f"<span>{snapshot['active']} active · {snapshot['waiting']} waiting · "
        f"{snapshot['free']} free / {snapshot['capacity']} slots</span>"
        "</div>"
        "<table class='admin-queue-table'>"
        "<thead><tr><th>State</th><th>Token</th><th>Position</th><th>Admitted</th><th>Started</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody>"
        "</table>"
        f"<div class='admin-queue-note'>Refreshed every {QUEUE_ADMIN_TIMER_SECONDS}s from local admission state.</div>"
        "</div>"
    )


def _public_queue_admit():
    global _queue_waiting, _queue_next_id
    with _queue_lock:
        if _queue_active + _queue_waiting >= PUBLIC_QUEUE_SIZE:
            return None, f"Queue full — all {PUBLIC_QUEUE_SIZE} public slots are admitted. Try again in a few minutes."
        _queue_next_id += 1
        token = f"public-{int(time.time())}-{_queue_next_id}"
        _queue_waiting += 1
        _queue_tokens[token] = "waiting"
        _queue_records[token] = {
            "token": token,
            "sequence": _queue_next_id,
            "position": _queue_waiting,
            "state": "waiting",
            "admitted_at": time.time(),
            "started_at": None,
        }
        position = _queue_waiting
        jobs_ahead = max(0, _queue_active + position - 1)
    noun = "job" if jobs_ahead == 1 else "jobs"
    return token, f"Queue slot reserved. You are waiting behind {jobs_ahead} public {noun}."


def _public_queue_mark_started(token):
    global _queue_active, _queue_waiting
    if not token:
        return False
    with _queue_lock:
        if _queue_tokens.get(token) != "waiting":
            return False
        _queue_tokens[token] = "active"
        if token in _queue_records:
            _queue_records[token]["state"] = "active"
            _queue_records[token]["started_at"] = time.time()
        _queue_waiting = max(0, _queue_waiting - 1)
        _queue_active += 1
    return True


def _public_queue_mark_finished(token):
    global _queue_active, _queue_waiting
    if not token:
        return
    with _queue_lock:
        state = _queue_tokens.pop(token, None)
        _queue_records.pop(token, None)
        if state == "active":
            _queue_active = max(0, _queue_active - 1)
        elif state == "waiting":
            _queue_waiting = max(0, _queue_waiting - 1)


def _public_queue_button_update():
    if not _is_public_mode():
        return gr.update(interactive=True)
    return gr.update(interactive=not _public_queue_is_full())


def _queue_status_tick():
    return _public_queue_status_html(), _public_queue_button_update()


def _queue_admin_tick():
    return _public_queue_admin_html()


def _admit_generate_click():
    if not _is_public_mode():
        return "", "Starting local generation.", _public_queue_status_html(), gr.update(interactive=True)
    token, message = _public_queue_admit()
    return token or "", message, _public_queue_status_html(), _public_queue_button_update()


def _build_prompt_payload(prompt_text, use_style, aesthetics, lighting, medium, color_palette,
                          use_composition, canvas, background, layout, elements_text):
    prompt_text = (prompt_text or "").strip()
    payload = {"high_level_description": prompt_text}
    if use_style:
        style = {}
        if (aesthetics or "").strip():
            style["aesthetics"] = aesthetics.strip()
        if (lighting or "").strip():
            style["lighting"] = lighting.strip()
        if (medium or "").strip():
            style["medium"] = medium.strip()
        colors = [part.strip() for part in (color_palette or "").split(",") if part.strip()]
        if colors:
            style["color_palette"] = colors
        if style:
            payload["style_description"] = style
    if use_composition:
        composition = {}
        if (canvas or "").strip():
            composition["canvas"] = canvas.strip()
        if (background or "").strip():
            composition["background"] = background.strip()
        if (layout or "").strip():
            composition["layout"] = layout.strip()
        elements = []
        for raw in (elements_text or "").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            if "|" in raw:
                kind, desc = raw.split("|", 1)
                kind = kind.strip() or "obj"
                desc = desc.strip()
            else:
                kind, desc = "obj", raw
            if desc:
                elements.append({"type": kind, "desc": desc})
        if elements:
            composition["elements"] = elements
        if composition:
            payload["compositional_deconstruction"] = composition
    if len(payload) == 1:
        return json.dumps({"prompt": prompt_text})
    return json.dumps(payload)


def _preset_choices(presets):
    return [PRESET_PLACEHOLDER] + [label for label, _ in presets]


def _advanced_choices():
    return [item["label"] for item in ADVANCED_RUN_PRESETS]


def _preset_value(presets, selected):
    if selected == PRESET_PLACEHOLDER:
        return ""
    for label, value in presets:
        if selected in (label, value):
            return value
    return selected or ""


def _advanced_preset_by_key(key):
    for item in ADVANCED_RUN_PRESETS:
        if key in (item["key"], item["label"]):
            return item
    return ADVANCED_RUN_PRESETS[0]


def _apply_text_preset(selected):
    return _preset_value(PROMPT_PRESETS, selected)


def _apply_style_preset(selected, preset_name):
    value = _preset_value(preset_name, selected)
    return bool(value), value


def _apply_composition_preset(selected, preset_name):
    value = _preset_value(preset_name, selected)
    return bool(value), value


def _apply_aesthetics_preset(selected):
    return _apply_style_preset(selected, AESTHETICS_PRESETS)


def _apply_lighting_preset(selected):
    return _apply_style_preset(selected, LIGHTING_PRESETS)


def _apply_medium_preset(selected):
    return _apply_style_preset(selected, MEDIUM_PRESETS)


def _apply_color_palette_preset(selected):
    return _apply_style_preset(selected, COLOR_PALETTE_PRESETS)


def _apply_canvas_preset(selected):
    return _apply_composition_preset(selected, CANVAS_PRESETS)


def _apply_background_preset(selected):
    return _apply_composition_preset(selected, BACKGROUND_PRESETS)


def _apply_layout_preset(selected):
    return _apply_composition_preset(selected, LAYOUT_PRESETS)


def _apply_elements_preset(selected):
    return _apply_composition_preset(selected, ELEMENTS_PRESETS)


def _apply_advanced_preset(selected):
    item = _advanced_preset_by_key(selected)
    preset_name = PUBLIC_PRESET
    width = int(item["width"])
    height = int(item["height"])
    steps = int(item["steps"])
    if _is_public_mode():
        width = _clamp_public_dimension(width, PUBLIC_MIN_WIDTH, PUBLIC_MAX_WIDTH)
        height = _clamp_public_dimension(height, PUBLIC_MIN_HEIGHT, PUBLIC_MAX_HEIGHT)
        steps = _clamp_public_steps(steps)
    return int(item["seed"]), preset_name, width, height, steps


def _randomize_form():
    prompt_text = random.choice(PROMPT_PRESETS)[1]
    aesthetics = random.choice(AESTHETICS_PRESETS)[1]
    lighting = random.choice(LIGHTING_PRESETS)[1]
    medium = random.choice(MEDIUM_PRESETS)[1]
    color_palette = random.choice(COLOR_PALETTE_PRESETS)[1]
    canvas = random.choice(CANVAS_PRESETS)[1]
    background = random.choice(BACKGROUND_PRESETS)[1]
    layout = random.choice(LAYOUT_PRESETS)[1]
    elements_text = random.choice(ELEMENTS_PRESETS)[1]
    seed, preset_name, width, height, steps = _apply_advanced_preset(
        random.choice(ADVANCED_RUN_PRESETS)["key"]
    )
    seed = random.randint(1, 999999)
    return (
        prompt_text,
        seed,
        preset_name,
        width,
        height,
        steps,
        True,
        aesthetics,
        lighting,
        medium,
        color_palette,
        True,
        canvas,
        background,
        layout,
        elements_text,
    )


def _assert_nf4_available():
    try:
        w = mx.random.normal((64, 64)).astype(mx.float16)
        q = mx.quantize(w, bits=4, group_size=64, mode="nf4")
        mx.eval(q[0])
    except Exception as e:
        raise RuntimeError(
            "NF4 support is not active in the current MLX install. Reinstall the "
            "NF4 fork last: pip install --force-reinstall --no-deps "
            "git+https://github.com/lyonsno/mlx.git@nf4"
        ) from e


def _write_run_receipt(img, prompt, seed, width, height, preset, num_steps,
                       sampling_time, active_gb, sampling_peak_gb, total_peak_gb):
    out_dir = os.path.join(os.path.dirname(__file__), "evidence", "live_runs")
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    stem = f"{stamp}_{ROUTE_ID}_{int(width)}x{int(height)}_seed{int(seed)}"
    image_path = os.path.join(out_dir, f"{stem}.png")
    receipt_path = os.path.join(out_dir, f"{stem}.json")

    img.save(image_path)
    with open(image_path, "rb") as fp:
        image_sha256 = hashlib.sha256(fp.read()).hexdigest()

    receipt = {
        "route": ROUTE_ID,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "quant_format": QUANT_FORMAT,
        "backend": BACKEND,
        "not_routes": ["MFLUX", "GGUF/stable-diffusion.cpp", "PyTorch/MPS", "remote GPU"],
        "host": os.uname().nodename,
        "public_mode": _is_public_mode(),
        "prompt": prompt,
        "seed": int(seed),
        "resolution": f"{int(width)}x{int(height)}",
        "steps": int(num_steps),
        "preset": preset,
        "sampling_time_s": round(float(sampling_time), 2),
        "seconds_per_step": round(float(sampling_time) / int(num_steps), 2),
        "memory_active_gb": round(float(active_gb), 2),
        "memory_sampling_peak_gb": round(float(sampling_peak_gb), 2),
        "memory_total_peak_gb": round(float(total_peak_gb), 2),
        "output": os.path.abspath(image_path),
        "output_sha256": image_sha256,
    }
    with open(receipt_path, "w") as fp:
        json.dump(receipt, fp, indent=2)
        fp.write("\n")
    return image_path, receipt_path, image_sha256


def _public_output_paths(seed, width, height):
    out_dir = os.path.join(os.path.dirname(__file__), "evidence", "live_runs")
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    stem = f"{stamp}_{ROUTE_ID}_{int(width)}x{int(height)}_seed{int(seed)}"
    return os.path.join(out_dir, f"{stem}.png"), os.path.join(out_dir, f"{stem}.json")


def _append_request_log(event):
    os.makedirs(os.path.dirname(REQUEST_LOG_PATH), exist_ok=True)
    payload = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **event}
    with open(REQUEST_LOG_PATH, "a") as fp:
        fp.write(json.dumps(payload, sort_keys=True) + "\n")


def _load_gallery_manifest():
    try:
        with open(GALLERY_MANIFEST_PATH) as fp:
            data = json.load(fp)
    except FileNotFoundError:
        data = {}
    discovered = [_rel_path(path) for path in _discover_generated_images()]
    promoted = list(dict.fromkeys(data.get("promoted", [])))
    shown = list(dict.fromkeys(data.get("shown", [])))
    hidden = list(dict.fromkeys(data.get("hidden", [])))
    hidden = list(dict.fromkeys(hidden + [
        rel for rel in discovered
        if rel not in promoted and rel not in shown and rel not in hidden
    ]))
    demoted = list(dict.fromkeys(data.get("demoted", [])))
    demoted = list(dict.fromkeys(demoted + [
        rel for rel in discovered
        if rel not in promoted and rel not in demoted
    ]))
    return {
        "promoted": promoted,
        "shown": shown,
        "demoted": demoted,
        "hidden": hidden,
    }


def _save_gallery_manifest(data):
    os.makedirs(os.path.dirname(GALLERY_MANIFEST_PATH), exist_ok=True)
    with open(GALLERY_MANIFEST_PATH, "w") as fp:
        json.dump(data, fp, indent=2)
        fp.write("\n")


def _rel_path(path):
    try:
        return os.path.relpath(path, os.path.dirname(__file__))
    except ValueError:
        return path


def _image_caption(path):
    rel = _rel_path(path)
    receipt = path[:-4] + ".json" if path.endswith(".png") else ""
    if receipt and os.path.exists(receipt):
        try:
            with open(receipt) as fp:
                data = json.load(fp)
            return f"{os.path.basename(path)} — {data.get('resolution', '?')} / {data.get('steps', '?')} steps / seed {data.get('seed', '?')}"
        except Exception:
            pass
    return rel


def _image_short_caption(path):
    receipt = path[:-4] + ".json" if path.endswith(".png") else ""
    if receipt and os.path.exists(receipt):
        try:
            with open(receipt) as fp:
                data = json.load(fp)
            return f"{data.get('resolution', '?')} / {data.get('steps', '?')} steps / seed {data.get('seed', '?')}"
        except Exception:
            pass
    name = os.path.basename(path)
    return name if len(name) <= 34 else name[:31] + "..."


def _discover_generated_images():
    seen = set()
    items = []
    for rel in PUBLIC_GALLERY_FILES:
        path = os.path.join(os.path.dirname(__file__), rel)
        if os.path.exists(path) and path not in seen:
            seen.add(path)
            items.append(path)
    for pattern in [
        os.path.join(os.path.dirname(__file__), "evidence", "live_runs", "*.png"),
        os.path.join(os.path.dirname(__file__), "evidence", "matrix", "nf4_*.png"),
        os.path.join(os.path.dirname(__file__), "evidence", "comparison", "nf4_*.png"),
        os.path.join(os.path.dirname(__file__), "evidence", "nf4_*.png"),
    ]:
        for path in sorted(glob.glob(pattern), reverse=True):
            if path not in seen:
                seen.add(path)
                items.append(path)
    return items


def _manifest_path_set(manifest, key):
    return {
        os.path.normpath(os.path.join(os.path.dirname(__file__), rel))
        for rel in manifest.get(key, [])
    }


def _visible_generated_images():
    manifest = _load_gallery_manifest()
    hidden = _manifest_path_set(manifest, "hidden")
    return [path for path in _discover_generated_images() if os.path.normpath(path) not in hidden]


def _hidden_generated_images():
    manifest = _load_gallery_manifest()
    items = []
    for rel in manifest.get("hidden", []):
        path = os.path.normpath(os.path.join(os.path.dirname(__file__), rel))
        if os.path.exists(path):
            items.append(path)
    return items


def _featured_gallery_paths():
    manifest = _load_gallery_manifest()
    demoted = _manifest_path_set(manifest, "demoted")
    hidden = _manifest_path_set(manifest, "hidden")
    promoted = []
    for rel in manifest["promoted"]:
        path = os.path.normpath(os.path.join(os.path.dirname(__file__), rel))
        if os.path.exists(path) and path not in demoted and path not in hidden:
            promoted.append(path)
    if promoted:
        return promoted
    return []


def _file_url(path):
    return "/gradio_api/file=" + urllib.parse.quote(os.path.abspath(path))


def _gallery_value(paths):
    items = []
    for path in paths:
        if not os.path.exists(path):
            continue
        caption = html.escape(_image_caption(path))
        url = _file_url(path)
        items.append(
            "<figure style='margin:0;min-width:0;'>"
            f"<img src='{url}' alt='{caption}' "
            "style='width:100%;aspect-ratio:1/1;object-fit:cover;border-radius:6px;border:1px solid #ddd;' />"
            f"<figcaption style='font-size:11px;line-height:1.25;margin-top:4px;color:#555;overflow-wrap:anywhere;'>{caption}</figcaption>"
            "</figure>"
        )
    if not items:
        return "<div style='color:#666;font-size:13px;'>No generated images found yet.</div>"
    return (
        "<div style='display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));"
        "gap:10px;align-items:start;'>"
        + "".join(items)
        + "</div>"
    )


def _featured_gallery_value(paths):
    if not paths:
        return "<div style='color:#666;font-size:13px;'>No promoted images yet.</div>"
    return _gallery_value(paths)


def _gallery_bin_value(paths):
    return [path for path in paths if os.path.exists(path)]


def _gallery_tile_html(path):
    caption = html.escape(_image_short_caption(path))
    url = _file_url(path)
    return (
        "<div style='width:100%;min-width:0;'>"
        f"<img src='{url}' alt='{caption}' "
        "style='width:100%;aspect-ratio:1/1;object-fit:cover;border-radius:6px;border:1px solid #ddd;' />"
        "</div>"
    )


def _gallery_dataset_samples(paths):
    return [[_gallery_tile_html(path)] for path in paths if os.path.exists(path)]


def _gallery_dataset_labels(paths):
    return ["" for path in paths if os.path.exists(path)]


def _gallery_dataset_update(paths):
    return gr.update(
        samples=_gallery_dataset_samples(paths),
        sample_labels=_gallery_dataset_labels(paths),
        value=None,
    )


def _gallery_slot_paths(paths):
    values = [path for path in paths[:GALLERY_SLOT_COUNT] if os.path.exists(path)]
    return values + [None] * (GALLERY_SLOT_COUNT - len(values))


def _gallery_slot_html(path, selected=False):
    if not path or not os.path.exists(path):
        return ""
    caption = html.escape(_image_short_caption(path))
    url = _file_url(path)
    selected_class = " selected" if selected else ""
    border = "#ff6b1a" if selected else "#4a4b52"
    shadow = "0 0 0 2px #ff6b1a" if selected else "none"
    return (
        f"<div class='gallery-slot{selected_class}' title='{caption}' "
        "style='width:100%;height:112px;display:flex;align-items:center;justify-content:center;"
        "padding:6px;border-radius:6px;background:#25262b;cursor:pointer;overflow:hidden;"
        f"border:2px solid {border};box-shadow:{shadow};'>"
        f"<img src='{url}' alt='{caption}' "
        "style='max-width:100%;max-height:100%;object-fit:contain;display:block;' />"
        "</div>"
    )


def _gallery_slot_updates(paths, selected_rel=None):
    return [
        gr.update(
            value=_gallery_slot_html(path, selected_rel and _rel_path(path) == selected_rel),
            visible=bool(path),
        )
        for path in _gallery_slot_paths(paths)
    ]


def _selected_path_by_index(paths, index):
    try:
        index = int(index)
    except Exception:
        return None
    if index < 0 or index >= len(paths):
        return None
    return _rel_path(paths[index])


def _selected_gallery_path(paths, evt):
    index = _selected_gallery_index(evt)
    if index is None or index < 0 or index >= len(paths):
        return None
    return _rel_path(paths[index])


def _selected_gallery_index(evt):
    try:
        index = evt.index[0] if isinstance(evt.index, (tuple, list)) else evt.index
        return int(index)
    except Exception:
        return None


def _gallery_choices():
    manifest = _load_gallery_manifest()
    hidden = _manifest_path_set(manifest, "hidden")
    choices = []
    for path in _discover_generated_images():
        label = _image_caption(path)
        if os.path.normpath(path) in hidden:
            label = f"[hidden] {label}"
        choices.append((label, _rel_path(path)))
    return choices


def _featured_wait_image(index=None):
    paths = _featured_gallery_paths()
    if not paths:
        return None
    if index is None:
        index = int(time.time() // GALLERY_ROTATE_SECONDS)
    return paths[int(index) % len(paths)]


def _featured_wait_output_image(index=None):
    path = _featured_wait_image(index)
    if not path:
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def _rotate_featured_wait_image(index):
    paths = _featured_gallery_paths()
    if not paths:
        return None, 0
    index = (int(index or 0) + 1) % len(paths)
    return paths[index], index


def _gallery_refresh_values(status=None):
    visible = _visible_generated_images()
    hidden = _hidden_generated_images()
    return (
        _featured_gallery_value(_featured_gallery_paths()),
        _gallery_value(visible),
        *_gallery_slot_updates(visible),
        *_gallery_slot_updates(hidden),
        None,
        None,
        status or f"{len(visible)} showing / {len(hidden)} hidden.",
    )


def _refresh_gallery():
    return _gallery_refresh_values()


def _select_visible_gallery(evt: gr.SelectData):
    selected_index = _selected_gallery_index(evt)
    selected = _selected_gallery_path(_visible_generated_images(), evt)
    if not selected:
        return None, None, gr.update(selected_index=None), gr.update(selected_index=None), "No showing image selected."
    return (
        selected,
        None,
        gr.update(selected_index=selected_index),
        gr.update(selected_index=None),
        f"Selected showing image: {_image_short_caption(os.path.join(os.path.dirname(__file__), selected))}",
    )


def _select_hidden_gallery(evt: gr.SelectData):
    selected_index = _selected_gallery_index(evt)
    selected = _selected_gallery_path(_hidden_generated_images(), evt)
    if not selected:
        return None, None, gr.update(selected_index=None), gr.update(selected_index=None), "No hidden image selected."
    return (
        None,
        selected,
        gr.update(selected_index=None),
        gr.update(selected_index=selected_index),
        f"Selected hidden image: {_image_short_caption(os.path.join(os.path.dirname(__file__), selected))}",
    )


def _select_visible_dataset(selected_index):
    try:
        index = int(selected_index)
    except Exception:
        return None, None, "No showing image selected."
    selected = _selected_gallery_path(_visible_generated_images(), type("Evt", (), {"index": index})())
    if not selected:
        return None, None, "No showing image selected."
    return selected, None, f"Selected showing image: {_image_short_caption(os.path.join(os.path.dirname(__file__), selected))}"


def _select_hidden_dataset(selected_index):
    try:
        index = int(selected_index)
    except Exception:
        return None, None, "No hidden image selected."
    selected = _selected_gallery_path(_hidden_generated_images(), type("Evt", (), {"index": index})())
    if not selected:
        return None, None, "No hidden image selected."
    return None, selected, f"Selected hidden image: {_image_short_caption(os.path.join(os.path.dirname(__file__), selected))}"


def _select_visible_slot(slot_index):
    selected = _selected_path_by_index(_visible_generated_images(), slot_index)
    if not selected:
        return (
            *_gallery_slot_updates(_visible_generated_images()),
            *_gallery_slot_updates(_hidden_generated_images()),
            None,
            None,
            "No showing image selected.",
        )
    path = os.path.join(os.path.dirname(__file__), selected)
    return (
        *_gallery_slot_updates(_visible_generated_images(), selected),
        *_gallery_slot_updates(_hidden_generated_images()),
        selected,
        None,
        f"Selected showing image: {_image_short_caption(path)}",
    )


def _select_hidden_slot(slot_index):
    selected = _selected_path_by_index(_hidden_generated_images(), slot_index)
    if not selected:
        return (
            *_gallery_slot_updates(_visible_generated_images()),
            *_gallery_slot_updates(_hidden_generated_images()),
            None,
            None,
            "No hidden image selected.",
        )
    path = os.path.join(os.path.dirname(__file__), selected)
    return (
        *_gallery_slot_updates(_visible_generated_images()),
        *_gallery_slot_updates(_hidden_generated_images(), selected),
        None,
        selected,
        f"Selected hidden image: {_image_short_caption(path)}",
    )


def _promote_gallery_item(selected):
    if not selected:
        return _gallery_refresh_values("Select a showing image before promoting.")
    manifest = _load_gallery_manifest()
    selected = _rel_path(os.path.join(os.path.dirname(__file__), selected))
    manifest["demoted"] = [p for p in manifest["demoted"] if p != selected]
    manifest["hidden"] = [p for p in manifest["hidden"] if p != selected]
    manifest["shown"] = [selected] + [p for p in manifest["shown"] if p != selected]
    manifest["promoted"] = [selected] + [p for p in manifest["promoted"] if p != selected]
    _save_gallery_manifest(manifest)
    return _gallery_refresh_values(f"Promoted {_image_short_caption(os.path.join(os.path.dirname(__file__), selected))}.")


def _demote_gallery_item(selected):
    if not selected:
        return _gallery_refresh_values("Select a showing image before demoting.")
    manifest = _load_gallery_manifest()
    selected = _rel_path(os.path.join(os.path.dirname(__file__), selected))
    manifest["promoted"] = [p for p in manifest["promoted"] if p != selected]
    manifest["shown"] = [selected] + [p for p in manifest["shown"] if p != selected]
    manifest["demoted"] = [selected] + [p for p in manifest["demoted"] if p != selected]
    _save_gallery_manifest(manifest)
    return _gallery_refresh_values(f"Demoted {_image_short_caption(os.path.join(os.path.dirname(__file__), selected))}.")


def _hide_gallery_item(selected):
    if not selected:
        return _gallery_refresh_values("Select a showing image before hiding.")
    manifest = _load_gallery_manifest()
    selected = _rel_path(os.path.join(os.path.dirname(__file__), selected))
    manifest["promoted"] = [path for path in manifest["promoted"] if path != selected]
    manifest["shown"] = [path for path in manifest["shown"] if path != selected]
    manifest["demoted"] = [selected] + [path for path in manifest["demoted"] if path != selected]
    manifest["hidden"] = [selected] + [path for path in manifest["hidden"] if path != selected]
    _save_gallery_manifest(manifest)
    return _gallery_refresh_values(f"Hidden {_image_short_caption(os.path.join(os.path.dirname(__file__), selected))}.")


def _show_gallery_item(selected):
    if not selected:
        return _gallery_refresh_values("Select a hidden image before showing.")
    manifest = _load_gallery_manifest()
    selected = _rel_path(os.path.join(os.path.dirname(__file__), selected))
    manifest["hidden"] = [path for path in manifest["hidden"] if path != selected]
    manifest["shown"] = [selected] + [path for path in manifest["shown"] if path != selected]
    _save_gallery_manifest(manifest)
    return _gallery_refresh_values(f"Showing {_image_short_caption(os.path.join(os.path.dirname(__file__), selected))}.")


def _generate_public_subprocess(prompt, seed, width, height, preset_name, steps):
    """Run public jobs through generate.py so memory matches the proven CLI route."""
    output_path, receipt_path = _public_output_paths(seed, width, height)
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "generate.py"),
        "--prompt", prompt,
        "--output", output_path,
        "--height", str(int(height)),
        "--width", str(int(width)),
        "--seed", str(int(seed)),
        "--preset", preset_name,
        "--steps", str(int(steps)),
        "--receipt", receipt_path,
    ]
    env = os.environ.copy()
    env.setdefault("HF_HUB_OFFLINE", "1")
    t0 = time.perf_counter()

    def running_status(elapsed, lines):
        detail = "\n".join(lines) if lines else "Waiting for generate.py output..."
        return (
            f"Running {int(width)}×{int(height)} / {int(steps)} steps on the 16 GB M2 Pro path\n"
            f"Elapsed: {elapsed:.0f}s\n\n{detail}"
        )

    proc = subprocess.Popen(
        cmd,
        cwd=os.path.dirname(__file__),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    lines = []
    assert proc.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    next_heartbeat = t0
    while proc.poll() is None:
        now = time.perf_counter()
        timeout = max(0.0, min(PUBLIC_STATUS_HEARTBEAT_SECONDS, next_heartbeat - now))
        events = selector.select(timeout=timeout)
        emitted = False
        for key, _ in events:
            line = key.fileobj.readline()
            if line == "":
                continue
            line = line.rstrip()
            if line:
                lines.append(line)
                lines = lines[-8:]
            elapsed = time.perf_counter() - t0
            yield _featured_wait_output_image(), running_status(elapsed, lines)
            next_heartbeat = time.perf_counter() + PUBLIC_STATUS_HEARTBEAT_SECONDS
            emitted = True
        if not emitted and time.perf_counter() >= next_heartbeat:
            elapsed = time.perf_counter() - t0
            yield _featured_wait_output_image(), running_status(elapsed, lines)
            next_heartbeat = time.perf_counter() + PUBLIC_STATUS_HEARTBEAT_SECONDS
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            lines.append(line)
            lines = lines[-8:]
            elapsed = time.perf_counter() - t0
            yield _featured_wait_output_image(), running_status(elapsed, lines)
    returncode = proc.wait()
    wall_time = time.perf_counter() - t0
    if returncode != 0:
        detail = "\n".join(lines).strip()
        raise RuntimeError(
            f"generate.py failed with exit {returncode} after {wall_time:.0f}s\n{detail}"
        )

    with open(output_path, "rb") as fp:
        image_sha256 = hashlib.sha256(fp.read()).hexdigest()
    try:
        with open(receipt_path) as fp:
            receipt = json.load(fp)
    except FileNotFoundError:
        receipt = {}
    receipt.update({
        "route": ROUTE_ID,
        "model": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "quant_format": QUANT_FORMAT,
        "backend": BACKEND,
        "not_routes": ["MFLUX", "GGUF/stable-diffusion.cpp", "PyTorch/MPS", "remote GPU"],
        "host": os.uname().nodename,
        "public_mode": True,
        "public_server_route": "gradio-subprocess-generate.py",
        "wall_time_s": round(wall_time, 2),
        "output": os.path.abspath(output_path),
        "output_sha256": image_sha256,
    })
    with open(receipt_path, "w") as fp:
        json.dump(receipt, fp, indent=2)
        fp.write("\n")
    image = Image.open(output_path).convert("RGB")
    status = (
        f"{int(width)}×{int(height)} | {int(steps)} steps | "
        f"{wall_time:.0f}s wall | seed {int(seed)}\n"
        f"Route: {ROUTE_ID} via generate.py subprocess | model {MODEL_REVISION[:12]} | "
        f"sha256 {image_sha256[:12]}\n"
        f"Receipt: {receipt_path}"
    )
    yield image, status


def _load_models(progress=gr.Progress()):
    """Load all models once."""
    if "loaded" in _state:
        return

    _assert_nf4_available()
    token = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
    model_id = MODEL_ID

    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    from mlx_vlm.models.qwen3_vl.config import ModelConfig, TextConfig, VisionConfig
    from mlx_vlm.models.qwen3_vl.qwen3_vl import Model as Qwen3VLModel

    # Tokenizer
    progress(0.05, desc="Loading tokenizer...")
    tok_dir = os.path.dirname(hf_hub_download(model_id, "tokenizer/tokenizer.json", revision=MODEL_REVISION, token=token))
    hf_hub_download(model_id, "tokenizer/chat_template.jinja", revision=MODEL_REVISION, token=token)
    hf_hub_download(model_id, "tokenizer/tokenizer_config.json", revision=MODEL_REVISION, token=token)
    _state["tokenizer"] = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)

    # Text encoder
    progress(0.1, desc="Loading text encoder (8.8B NF4)...")
    cfg_path = hf_hub_download(model_id, "text_encoder/config.json", revision=MODEL_REVISION, token=token)
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
    wp = hf_hub_download(model_id, "text_encoder/model.safetensors", revision=MODEL_REVISION, token=token)
    load_nf4_text_encoder(wp, text_model, verbose=False)
    _state["text_model"] = text_model

    # Transformers
    progress(0.4, desc="Loading conditional transformer (9.3B NF4)...")
    cf = hf_hub_download(model_id, "transformer/diffusion_pytorch_model.safetensors", revision=MODEL_REVISION, token=token)
    cond = Ideogram4Transformer()
    load_nf4_transformer(cf, cond, verbose=False)
    _state["cond_model"] = cond

    progress(0.6, desc="Loading unconditional transformer (9.3B NF4)...")
    uf = hf_hub_download(model_id, "unconditional_transformer/diffusion_pytorch_model.safetensors", revision=MODEL_REVISION, token=token)
    uncond = Ideogram4Transformer()
    load_nf4_transformer(uf, uncond, verbose=False)
    _state["uncond_model"] = uncond

    # VAE
    progress(0.8, desc="Loading VAE decoder...")
    vf = hf_hub_download(model_id, "vae/diffusion_pytorch_model.safetensors", revision=MODEL_REVISION, token=token)
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
    _state["model_revision"] = MODEL_REVISION
    _state["loaded"] = True
    progress(1.0, desc="Ready!")


def _guidance_for_steps(preset, num_steps):
    guidance = preset["guidance"]
    if len(guidance) == num_steps:
        return guidance
    if num_steps > len(guidance):
        return (7.0,) * max(0, num_steps - 3) + (3.0,) * min(3, num_steps)
    return guidance[:num_steps]


def generate(prompt_text, use_json, seed, preset_name, width, height, steps,
             use_style, aesthetics, lighting, medium, color_palette,
             use_composition, canvas, background, layout, elements_text,
             progress=gr.Progress()):
    """Generate an image from a text prompt."""
    global _last_gen_time

    # Rate limit
    with _rate_lock:
        now = time.time()
        elapsed = now - _last_gen_time
        if elapsed < _MIN_COOLDOWN and _last_gen_time > 0:
            wait = int(_MIN_COOLDOWN - elapsed)
            yield None, f"Rate limited — please wait {wait}s before generating again"
            return
        _last_gen_time = now

    prompt_text, use_json, preset_name, width, height, steps = _effective_request(
        prompt_text, use_json, preset_name, width, height, steps
    )
    if not prompt_text:
        yield _featured_wait_output_image(), "Enter a prompt before generating"
        return

    if _is_public_mode():
        request_event = {
            "event": "request_start",
            "prompt_text": prompt_text,
            "seed": int(seed),
            "requested_width": int(width),
            "requested_height": int(height),
            "effective_width": int(width),
            "effective_height": int(height),
            "steps": int(steps),
            "use_style": bool(use_style),
            "use_composition": bool(use_composition),
        }
        _append_request_log(request_event)
        yield _featured_wait_output_image(), (
            f"Queued on the 16 GB M2 Pro path: {int(width)}×{int(height)}, "
            f"{int(steps)} steps. Expected latency: up to {PUBLIC_EXPECTED_LATENCY} at 512×512 / 20 steps."
        )
        try:
            prompt = _build_prompt_payload(
                prompt_text,
                use_style,
                aesthetics,
                lighting,
                medium,
                color_palette,
                use_composition,
                canvas,
                background,
                layout,
                elements_text,
            )
            yield from _generate_public_subprocess(prompt, seed, width, height, preset_name, steps)
        except Exception as e:
            _append_request_log({"event": "request_error", "error": f"{type(e).__name__}: {e}"})
            yield _featured_wait_output_image(), f"{type(e).__name__}: {e}"
            return
        _append_request_log({"event": "request_complete", "seed": int(seed), "width": int(width), "height": int(height), "steps": int(steps)})
        return

    try:
        _assert_nf4_available()
    except RuntimeError as e:
        yield _featured_wait_output_image(), str(e)
        return

    _load_models(progress)

    # In JSON mode, pass through raw. Otherwise wrap plain text.
    if use_json:
        prompt = prompt_text
    else:
        prompt = _build_prompt_payload(
            prompt_text,
            use_style,
            aesthetics,
            lighting,
            medium,
            color_palette,
            use_composition,
            canvas,
            background,
            layout,
            elements_text,
        )

    preset = PRESETS[preset_name]
    num_steps = int(steps)
    guidance = _guidance_for_steps(preset, num_steps)

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

    # Preview every ~5 steps (disabled in public mode to save memory)
    previews_enabled = os.environ.get("NF4_NO_PREVIEW") != "1"
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
    image_path, receipt_path, image_sha256 = _write_run_receipt(
        img,
        prompt,
        seed,
        width,
        height,
        preset_name,
        num_steps,
        sampling_time,
        active,
        sampling_peak,
        total_peak,
    )
    info = (f"{int(width)}×{int(height)} | {num_steps} steps | {sampling_time:.0f}s sampling "
            f"({sampling_time/num_steps:.1f}s/step) | seed {int(seed)}\n"
            f"Memory: {active:.1f} GB active | {sampling_peak:.1f} GB sampling peak | "
            f"{total_peak:.1f} GB total peak (incl. previews)\n"
            f"Route: {ROUTE_ID} | model {MODEL_REVISION[:12]} | sha256 {image_sha256[:12]}\n"
            f"Receipt: {receipt_path}")

    yield img, info


def _generate_with_admission(admission_token, prompt_text, use_json, seed, preset_name, width, height, steps,
                             use_style, aesthetics, lighting, medium, color_palette,
                             use_composition, canvas, background, layout, elements_text,
                             progress=gr.Progress()):
    """Run generation after a fast public admission check has reserved a slot."""
    if not _is_public_mode():
        for image, status in generate(
            prompt_text, use_json, seed, preset_name, width, height, steps,
            use_style, aesthetics, lighting, medium, color_palette,
            use_composition, canvas, background, layout, elements_text,
            progress=progress,
        ):
            yield image, status, _public_queue_status_html(), gr.update(interactive=True)
        return

    if not admission_token or not _public_queue_mark_started(admission_token):
        yield (
            _featured_wait_output_image(),
            f"Queue full — all {PUBLIC_QUEUE_SIZE} public slots are admitted. Try again in a few minutes.",
            _public_queue_status_html(),
            _public_queue_button_update(),
        )
        return

    last = None
    try:
        for image, status in generate(
            prompt_text, use_json, seed, preset_name, width, height, steps,
            use_style, aesthetics, lighting, medium, color_palette,
            use_composition, canvas, background, layout, elements_text,
            progress=progress,
        ):
            last = (image, status)
            yield image, status, _public_queue_status_html(), _public_queue_button_update()
    finally:
        _public_queue_mark_finished(admission_token)

    if last is not None:
        yield last[0], last[1], _public_queue_status_html(), _public_queue_button_update()


# Build UI
_PUBLIC_MODE = _is_public_mode()

with gr.Blocks(title="Ideogram4 NF4 — Apple Silicon") as demo:

    # === Header ===
    public_line = (
        f"**Public demo mode:** one {PUBLIC_HARDWARE}, hard-locked to "
        f"at most {PUBLIC_MAX_STEPS} steps and capped at {PUBLIC_MAX_WIDTH}×{PUBLIC_MAX_HEIGHT}. Expected latency is "
        f"up to {PUBLIC_EXPECTED_LATENCY} per 512×512 sample; the public queue holds up to {PUBLIC_QUEUE_SIZE} jobs."
        if _PUBLIC_MODE else
        "**Local mode:** controls are open for local experimentation."
    )
    gr.Markdown(f"""
# Ideogram4 NF4 on Apple Silicon

**You are looking at a live demo running on one Mac.**
This is [Ideogram 4](https://ideogram.ai) — a 9.3B parameter state-of-the-art text-to-image model
with best-in-class text rendering — running through custom NF4 Metal kernels at 4-bit precision.

{public_line}

Route: **official bitsandbytes NF4 weights → MLX custom NF4 Metal kernels**.
Not MFLUX, not GGUF/stable-diffusion.cpp, not PyTorch/MPS, and not a remote GPU.

The 512×512 / 20-step route fits in **11.5 GB** of memory on the 16 GB M2 Pro receipt.
The FP8 comparison route needs about 28 GB.

<details>
<summary><b>What is NF4?</b></summary>

NF4 (NormalFloat4) is a 4-bit quantization format from the [QLoRA paper](https://arxiv.org/abs/2305.14314)
used by [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes). It places 16 quantization
levels at the quantiles of a normal distribution — optimal for neural network weights, which are roughly Gaussian.

This demo uses [Metal kernels for MLX](https://github.com/lyonsno/mlx/tree/nf4)
to load official bitsandbytes NF4 weights directly on Apple Silicon. No re-quantization,
no conversion — the same checkpoint files, half the memory of FP8.

</details>

<details>
<summary><b>Want to run it yourself?</b></summary>

The install path is intentionally short. If dependency hell still finds you, hit me up:
[github.com/lyonsno](https://github.com/lyonsno)

```
git clone https://github.com/lyonsno/mlx-ideogram4.git && cd mlx-ideogram4
pip install -e .
pip install --force-reinstall --no-deps git+https://github.com/lyonsno/mlx.git@nf4
hf auth login
python generate.py --prompt "a red cat on a blue couch" --output cat.png
```

The fork install goes last because mlx-vlm pulls stock MLX as a dependency.
`generate.py` will fail loud with an exact fix if NF4 isn't active.

Repo: [github.com/lyonsno/mlx-ideogram4](https://github.com/lyonsno/mlx-ideogram4)
NF4 MLX fork: [github.com/lyonsno/mlx/tree/nf4](https://github.com/lyonsno/mlx/tree/nf4)

Model weights are under [Ideogram's non-commercial license](https://huggingface.co/ideogram-ai/ideogram-4-nf4).

</details>

---
    """)

    # === Generator ===
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Row():
                prompt_preset = gr.Dropdown(
                    label="Prompt preset",
                    choices=_preset_choices(PROMPT_PRESETS),
                    value="Red cat on couch",
                )
                randomize_btn = gr.Button("Randomize All")
            prompt = gr.Textbox(
                label="Prompt",
                placeholder='a red cat sitting on a blue couch',
                lines=2,
                value='a red cat sitting on a blue couch',
            )
            use_json = gr.Checkbox(label="Advanced JSON mode", value=False,
                                   info="Edit raw JSON for style/layout control",
                                   interactive=not _PUBLIC_MODE,
                                   visible=not _PUBLIC_MODE)
            run_recipe = gr.Dropdown(
                label="Advanced run preset",
                choices=_advanced_choices(),
                value="512 balanced",
            )
            with gr.Row():
                seed = gr.Number(label="Seed", value=42, precision=0)
                preset = gr.Dropdown(
                    label="Preset",
                    choices=[PUBLIC_PRESET] if _PUBLIC_MODE else list(PRESETS.keys()),
                    value=PUBLIC_PRESET,
                    interactive=not _PUBLIC_MODE,
                )
                steps = gr.Slider(
                    PUBLIC_MIN_STEPS if _PUBLIC_MODE else 1,
                    PUBLIC_MAX_STEPS if _PUBLIC_MODE else 48,
                    value=PUBLIC_STEPS if _PUBLIC_MODE else PRESETS[PUBLIC_PRESET]["steps"],
                    step=1,
                    label="Steps",
                    interactive=True,
                )
            with gr.Row():
                if _PUBLIC_MODE:
                    width = gr.Slider(
                        PUBLIC_MIN_WIDTH,
                        PUBLIC_MAX_WIDTH,
                        value=PUBLIC_WIDTH,
                        step=16,
                        label="Width",
                        interactive=True,
                    )
                    height = gr.Slider(
                        PUBLIC_MIN_HEIGHT,
                        PUBLIC_MAX_HEIGHT,
                        value=PUBLIC_HEIGHT,
                        step=16,
                        label="Height",
                        interactive=True,
                    )
                else:
                    width = gr.Slider(256, 1024, value=PUBLIC_WIDTH, step=16, label="Width")
                    height = gr.Slider(256, 1024, value=PUBLIC_HEIGHT, step=16, label="Height")
            btn = gr.Button("Generate", variant="primary", size="lg")
            admission_token = gr.State("")
            queue_status = gr.HTML(
                value=_public_queue_status_html(),
                visible=_PUBLIC_MODE,
            )
            queue_timer = gr.Timer(QUEUE_STATUS_TIMER_SECONDS, active=_PUBLIC_MODE)
            gr.Markdown(
                f"*Expected latency: up to {PUBLIC_EXPECTED_LATENCY} at 512×512 / 20 steps on the "
                f"16 GB M2 Pro. The public queue holds up to {PUBLIC_QUEUE_SIZE} jobs; Generate only rejects when that queue is full.*"
                if _PUBLIC_MODE else
                "*Expected latency depends on hardware and preset.*"
            )
            with gr.Accordion("Structured prompt fields", open=False):
                use_style = gr.Checkbox(label="Use style fields", value=False)
                with gr.Row():
                    aesthetics_preset = gr.Dropdown(
                        label="Aesthetics preset",
                        choices=_preset_choices(AESTHETICS_PRESETS),
                        value=PRESET_PLACEHOLDER,
                        filterable=False,
                        interactive=True,
                    )
                    lighting_preset = gr.Dropdown(
                        label="Lighting preset",
                        choices=_preset_choices(LIGHTING_PRESETS),
                        value=PRESET_PLACEHOLDER,
                        filterable=False,
                        interactive=True,
                    )
                with gr.Row():
                    aesthetics = gr.Textbox(label="Aesthetics", value="", placeholder="minimal graphic design", interactive=True)
                    lighting = gr.Textbox(label="Lighting", value="", placeholder="flat, golden hour, studio", interactive=True)
                with gr.Row():
                    medium_preset = gr.Dropdown(
                        label="Medium preset",
                        choices=_preset_choices(MEDIUM_PRESETS),
                        value=PRESET_PLACEHOLDER,
                        filterable=False,
                        interactive=True,
                    )
                    color_palette_preset = gr.Dropdown(
                        label="Palette preset",
                        choices=_preset_choices(COLOR_PALETTE_PRESETS),
                        value=PRESET_PLACEHOLDER,
                        filterable=False,
                        interactive=True,
                    )
                with gr.Row():
                    medium = gr.Textbox(label="Medium", value="", placeholder="digital vector art, photo, poster", interactive=True)
                    color_palette = gr.Textbox(label="Color palette", value="", placeholder="#000000, #FFFFFF", interactive=True)
                use_composition = gr.Checkbox(label="Use composition fields", value=False)
                with gr.Row():
                    canvas_preset = gr.Dropdown(
                        label="Canvas preset",
                        choices=_preset_choices(CANVAS_PRESETS),
                        value=PRESET_PLACEHOLDER,
                        filterable=False,
                        interactive=True,
                    )
                    background_preset = gr.Dropdown(
                        label="Background preset",
                        choices=_preset_choices(BACKGROUND_PRESETS),
                        value=PRESET_PLACEHOLDER,
                        filterable=False,
                        interactive=True,
                    )
                canvas = gr.Textbox(label="Canvas", value="", placeholder="512 x 512 square canvas", interactive=True)
                background = gr.Textbox(label="Background", value="", placeholder="pure white", interactive=True)
                with gr.Row():
                    layout_preset = gr.Dropdown(
                        label="Layout preset",
                        choices=_preset_choices(LAYOUT_PRESETS),
                        value=PRESET_PLACEHOLDER,
                        filterable=False,
                        interactive=True,
                    )
                    elements_preset = gr.Dropdown(
                        label="Elements preset",
                        choices=_preset_choices(ELEMENTS_PRESETS),
                        value=PRESET_PLACEHOLDER,
                        filterable=False,
                        interactive=True,
                    )
                layout = gr.Textbox(label="Layout", value="", placeholder="centered single element", interactive=True)
                elements_text = gr.Textbox(
                    label="Elements",
                    value="",
                    lines=3,
                    placeholder="text | Large bold black letters NF4\nobj | Apple logo silhouette",
                    interactive=True,
                )

        with gr.Column(scale=1):
            output_image = gr.Image(
                value=_featured_wait_output_image(0),
                label="Output",
                type="pil",
                height=512,
            )
            info = gr.Textbox(label="Status", interactive=False,
                              value="Ready — pick a prompt and click Generate")
            featured_gallery = gr.HTML(
                value=_featured_gallery_value(_featured_gallery_paths()),
                label="Featured prior runs",
            )

    # === Gallery ===
    gr.Markdown("---\n### Generated with NF4")
    all_gallery = gr.HTML(
        value=_gallery_value(_visible_generated_images()),
        label="All generated images",
    )
    with gr.Accordion("Gallery visibility console", open=_PUBLIC_MODE, elem_id=GALLERY_CONSOLE_ELEM_ID):
        queue_admin = gr.HTML(
            value=_public_queue_admin_html(),
            label="Admin queue",
        )
        queue_admin_timer = gr.Timer(QUEUE_ADMIN_TIMER_SECONDS, active=_PUBLIC_MODE)
        selected_visible_image = gr.State(None)
        selected_hidden_image = gr.State(None)
        visible_slot_tiles = []
        hidden_slot_tiles = []
        visible_slot_paths = _gallery_slot_paths(_visible_generated_images())
        hidden_slot_paths = _gallery_slot_paths(_hidden_generated_images())
        with gr.Row():
            with gr.Column():
                gr.Markdown("#### Showing")
                for row in range(10):
                    with gr.Row():
                        for col in range(4):
                            slot = row * 4 + col
                            tile = gr.HTML(
                                value=_gallery_slot_html(visible_slot_paths[slot]),
                                show_label=False,
                                visible=bool(visible_slot_paths[slot]),
                            )
                            visible_slot_tiles.append(tile)
            with gr.Column():
                gr.Markdown("#### Hidden")
                for row in range(10):
                    with gr.Row():
                        for col in range(4):
                            slot = row * 4 + col
                            tile = gr.HTML(
                                value=_gallery_slot_html(hidden_slot_paths[slot]),
                                show_label=False,
                                visible=bool(hidden_slot_paths[slot]),
                            )
                            hidden_slot_tiles.append(tile)
        gallery_status = gr.Textbox(
            label="Gallery status",
            interactive=False,
            value=f"{len(_visible_generated_images())} showing / {len(_hidden_generated_images())} hidden.",
        )
        with gr.Row():
            refresh_gallery_btn = gr.Button("Refresh Gallery")
            promote_btn = gr.Button("Promote")
            demote_btn = gr.Button("Demote")
        with gr.Row():
            hide_gallery_btn = gr.Button("Hide from Gallery")
            show_gallery_btn = gr.Button("Show in Gallery")

    # === Performance table ===
    gr.Markdown("""
---
### Performance (uncontended, M4 Max 128 GB)

| | NF4/MLX (this) | MFLUX FP8 |
|---|---|---|
| **512×512 / 20 steps** | **3.3s/step, 67s sampling** | 3.3s/step, 66s sampling |
| **Peak memory** | **11.5 GB** | **28.1 GB** |
| **Fits 16 GB Mac?** | **Yes** | No |

Same speed. 2.4× less memory.
    """)

    queue_timer.tick(
        fn=_queue_status_tick,
        inputs=[],
        outputs=[queue_status, btn],
        queue=False,
        show_progress="hidden",
    )
    queue_admin_timer.tick(
        fn=_queue_admin_tick,
        inputs=[],
        outputs=[queue_admin],
        queue=False,
        show_progress="hidden",
    )
    admit_event = btn.click(
        fn=_admit_generate_click,
        inputs=[],
        outputs=[admission_token, info, queue_status, btn],
        queue=False,
        show_progress="hidden",
    )
    admit_event.success(
        fn=_generate_with_admission,
        inputs=[
            admission_token,
            prompt, use_json, seed, preset, width, height, steps,
            use_style, aesthetics, lighting, medium, color_palette,
            use_composition, canvas, background, layout, elements_text,
        ],
        outputs=[output_image, info, queue_status, btn],
        show_progress="minimal",
    )
    prompt_preset.change(
        fn=_apply_text_preset,
        inputs=[prompt_preset],
        outputs=[prompt],
        queue=False,
        show_progress="hidden",
    )
    run_recipe.change(
        fn=_apply_advanced_preset,
        inputs=[run_recipe],
        outputs=[seed, preset, width, height, steps],
        queue=False,
        show_progress="hidden",
    )
    aesthetics_preset.change(
        fn=_apply_aesthetics_preset,
        inputs=[aesthetics_preset],
        outputs=[use_style, aesthetics],
        queue=False,
        show_progress="hidden",
    )
    lighting_preset.change(
        fn=_apply_lighting_preset,
        inputs=[lighting_preset],
        outputs=[use_style, lighting],
        queue=False,
        show_progress="hidden",
    )
    medium_preset.change(
        fn=_apply_medium_preset,
        inputs=[medium_preset],
        outputs=[use_style, medium],
        queue=False,
        show_progress="hidden",
    )
    color_palette_preset.change(
        fn=_apply_color_palette_preset,
        inputs=[color_palette_preset],
        outputs=[use_style, color_palette],
        queue=False,
        show_progress="hidden",
    )
    canvas_preset.change(
        fn=_apply_canvas_preset,
        inputs=[canvas_preset],
        outputs=[use_composition, canvas],
        queue=False,
        show_progress="hidden",
    )
    background_preset.change(
        fn=_apply_background_preset,
        inputs=[background_preset],
        outputs=[use_composition, background],
        queue=False,
        show_progress="hidden",
    )
    layout_preset.change(
        fn=_apply_layout_preset,
        inputs=[layout_preset],
        outputs=[use_composition, layout],
        queue=False,
        show_progress="hidden",
    )
    elements_preset.change(
        fn=_apply_elements_preset,
        inputs=[elements_preset],
        outputs=[use_composition, elements_text],
        queue=False,
        show_progress="hidden",
    )
    randomize_btn.click(
        fn=_randomize_form,
        inputs=[],
        outputs=[
            prompt, seed, preset, width, height, steps,
            use_style, aesthetics, lighting, medium, color_palette,
            use_composition, canvas, background, layout, elements_text,
        ],
        queue=False,
        show_progress="hidden",
    )
    refresh_gallery_btn.click(
        fn=_refresh_gallery,
        inputs=[],
        outputs=[
            featured_gallery, all_gallery,
            *visible_slot_tiles, *hidden_slot_tiles,
            selected_visible_image, selected_hidden_image, gallery_status,
        ],
        queue=False,
        show_progress="hidden",
    )
    for slot, slot_tile in enumerate(visible_slot_tiles):
        slot_tile.click(
            fn=lambda slot=slot: _select_visible_slot(slot),
            inputs=[],
            outputs=[
                *visible_slot_tiles, *hidden_slot_tiles,
                selected_visible_image, selected_hidden_image, gallery_status,
            ],
            queue=False,
            show_progress="hidden",
        )
    for slot, slot_tile in enumerate(hidden_slot_tiles):
        slot_tile.click(
            fn=lambda slot=slot: _select_hidden_slot(slot),
            inputs=[],
            outputs=[
                *visible_slot_tiles, *hidden_slot_tiles,
                selected_visible_image, selected_hidden_image, gallery_status,
            ],
            queue=False,
            show_progress="hidden",
        )
    promote_btn.click(
        fn=_promote_gallery_item,
        inputs=[selected_visible_image],
        outputs=[
            featured_gallery, all_gallery,
            *visible_slot_tiles, *hidden_slot_tiles,
            selected_visible_image, selected_hidden_image, gallery_status,
        ],
        queue=False,
        show_progress="hidden",
    )
    demote_btn.click(
        fn=_demote_gallery_item,
        inputs=[selected_visible_image],
        outputs=[
            featured_gallery, all_gallery,
            *visible_slot_tiles, *hidden_slot_tiles,
            selected_visible_image, selected_hidden_image, gallery_status,
        ],
        queue=False,
        show_progress="hidden",
    )
    hide_gallery_btn.click(
        fn=_hide_gallery_item,
        inputs=[selected_visible_image],
        outputs=[
            featured_gallery, all_gallery,
            *visible_slot_tiles, *hidden_slot_tiles,
            selected_visible_image, selected_hidden_image, gallery_status,
        ],
        queue=False,
        show_progress="hidden",
    )
    show_gallery_btn.click(
        fn=_show_gallery_item,
        inputs=[selected_hidden_image],
        outputs=[
            featured_gallery, all_gallery,
            *visible_slot_tiles, *hidden_slot_tiles,
            selected_visible_image, selected_hidden_image, gallery_status,
        ],
        queue=False,
        show_progress="hidden",
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create public Gradio URL")
    parser.add_argument("--auth", type=str, default=None, help="user:password for basic auth")
    parser.add_argument("--public", action="store_true",
                        help=f"Public mode: queue={PUBLIC_QUEUE_SIZE}, 512x512, 20-step preset, no raw JSON, no previews")
    args = parser.parse_args()

    if args.public:
        # Lock down for public hosting:
        # - Queue size PUBLIC_QUEUE_SIZE (one active job, several waiting jobs)
        # - Disable previews (save memory on small boxes)
        # - Cap at 512x512 / 20 steps / no raw JSON in _effective_request()
        os.environ["NF4_PUBLIC_MODE"] = "1"
        os.environ["NF4_NO_PREVIEW"] = "1"
        demo.queue(max_size=PUBLIC_QUEUE_SIZE, default_concurrency_limit=1)
        print(f"PUBLIC MODE: queue={PUBLIC_QUEUE_SIZE}, <=512x512, <=20 steps, V4_DEFAULT_20, raw JSON disabled, previews disabled", flush=True)
    else:
        demo.queue(max_size=2)

    auth = None
    if args.auth:
        user, pw = args.auth.split(":", 1)
        auth = (user, pw)

    demo.launch(
        share=args.share,
        auth=auth,
        allowed_paths=[os.path.join(os.path.dirname(__file__), "evidence")],
        css=_gallery_console_visibility_css(_PUBLIC_MODE),
        head=_gallery_console_visibility_head(),
    )
