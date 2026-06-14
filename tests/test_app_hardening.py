import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class FakeComponent:
    instances = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        FakeComponent.instances.append(self)

    def click(self, *args, **kwargs):
        self.click_args = args
        self.click_kwargs = kwargs
        return None


class FakeContext(FakeComponent):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def queue(self, *args, **kwargs):
        self.queue_args = args
        self.queue_kwargs = kwargs
        return self

    def launch(self, *args, **kwargs):
        self.launch_args = args
        self.launch_kwargs = kwargs
        return None


def _component_class(name):
    return type(name, (FakeComponent,), {})


def install_fakes():
    FakeComponent.instances = []

    gradio = types.ModuleType("gradio")
    gradio.Blocks = type("Blocks", (FakeContext,), {})
    gradio.Row = type("Row", (FakeContext,), {})
    gradio.Column = type("Column", (FakeContext,), {})
    for name in [
        "Markdown",
        "Gallery",
        "Textbox",
        "Checkbox",
        "Number",
        "Dropdown",
        "Slider",
        "Button",
        "Image",
        "Examples",
    ]:
        setattr(gradio, name, _component_class(name))
    gradio.Progress = _component_class("Progress")
    gradio.themes = types.SimpleNamespace(Soft=lambda: object())
    sys.modules["gradio"] = gradio

    pil = types.ModuleType("PIL")
    pil_image = types.ModuleType("PIL.Image")
    pil_image.fromarray = lambda value: value
    pil.Image = pil_image
    sys.modules["PIL"] = pil
    sys.modules["PIL.Image"] = pil_image

    mlx = types.ModuleType("mlx")
    mlx_core = types.ModuleType("mlx.core")
    mlx.core = mlx_core
    sys.modules["mlx"] = mlx
    sys.modules["mlx.core"] = mlx_core

    scheduler = types.ModuleType("scheduler")
    scheduler.LogitNormalSchedule = object
    scheduler.make_step_intervals = lambda steps: list(range(steps + 1))
    sys.modules["scheduler"] = scheduler

    transformer = types.ModuleType("transformer")
    transformer.Ideogram4Transformer = object
    sys.modules["transformer"] = transformer

    load_weights = types.ModuleType("load_weights")
    load_weights.load_nf4_transformer = lambda *args, **kwargs: None
    sys.modules["load_weights"] = load_weights

    load_text_encoder = types.ModuleType("load_text_encoder")
    load_text_encoder.load_nf4_text_encoder = lambda *args, **kwargs: None
    sys.modules["load_text_encoder"] = load_text_encoder

    pipeline = types.ModuleType("pipeline")
    pipeline.build_inputs = lambda *args, **kwargs: {}
    pipeline.LATENT_SHIFT = 0.0
    pipeline.LATENT_SCALE = 1.0
    sys.modules["pipeline"] = pipeline

    vae = types.ModuleType("vae")
    vae.Decoder = object
    vae.decode_latents = lambda *args, **kwargs: None
    sys.modules["vae"] = vae

    generate = types.ModuleType("generate")
    generate._assert_nf4_available = lambda: None
    sys.modules["generate"] = generate


def load_app(public=False):
    install_fakes()
    sys.modules.pop("app", None)
    old_argv = sys.argv[:]
    old_path = sys.path[:]
    try:
        sys.argv = ["app.py"] + (["--public"] if public else [])
        sys.path.insert(0, str(ROOT))
        spec = importlib.util.spec_from_file_location("app", ROOT / "app.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["app"] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.argv = old_argv
        sys.path[:] = old_path


def components_named(name):
    return [c for c in FakeComponent.instances if type(c).__name__ == name]


class AppHardeningTests(unittest.TestCase):
    def test_normalize_request_caps_prompt_bad_preset_and_public_limits(self):
        app = load_app(public=True)
        long_prompt = "x" * (app.MAX_PROMPT_CHARS + 25)

        fallback_request = app._normalize_generation_request(
            prompt_text=long_prompt,
            use_json=False,
            seed=42,
            preset_name="NOT_A_PRESET",
            width=2048,
            height=2048,
            public_mode=True,
        )
        twenty_step_request = app._normalize_generation_request(
            prompt_text="ok",
            use_json=False,
            seed=42,
            preset_name="V4_DEFAULT_20",
            width=1024,
            height=768,
            public_mode=True,
        )

        self.assertEqual(app.MAX_PROMPT_CHARS, len(fallback_request.prompt_text))
        self.assertEqual("V4_TURBO_12", fallback_request.preset_name)
        self.assertEqual(1024, fallback_request.width)
        self.assertEqual(1024, fallback_request.height)
        self.assertEqual(app.PRESETS["V4_TURBO_12"], fallback_request.preset)
        self.assertEqual("V4_DEFAULT_20", twenty_step_request.preset_name)
        self.assertEqual(1024, twenty_step_request.width)
        self.assertEqual(768, twenty_step_request.height)
        self.assertEqual(app.PRESETS["V4_DEFAULT_20"], twenty_step_request.preset)

    def test_public_ui_limits_prompt_preset_and_resolution(self):
        app = load_app(public=True)

        textbox = components_named("Textbox")[0]
        dropdown = components_named("Dropdown")[0]
        sliders = components_named("Slider")

        self.assertEqual(app.MAX_PROMPT_CHARS, textbox.kwargs["max_length"])
        self.assertEqual(["V4_TURBO_12", "V4_DEFAULT_20"], dropdown.kwargs["choices"])
        self.assertEqual("V4_TURBO_12", dropdown.kwargs["value"])
        self.assertEqual(1024, sliders[0].args[1])
        self.assertEqual(1024, sliders[1].args[1])

    def test_cooldown_message_names_shared_server(self):
        app = load_app()

        message = app._cooldown_message(17)

        self.assertIn("server is cooling down", message)
        self.assertIn("shared cooldown", message)
        self.assertIn("17s", message)

    def test_gallery_displays_all_nf4_images(self):
        app = load_app()

        gallery = components_named("Gallery")[0]

        self.assertEqual(13, len(app._gallery_images))
        self.assertEqual(len(app._gallery_images), len(gallery.kwargs["value"]))

    def test_token_helper_uses_huggingface_resolution(self):
        app = load_app()

        self.assertTrue(hasattr(app, "_get_hf_token"))
        self.assertNotIn("~/.cache/huggingface/token", app._get_hf_token.__code__.co_consts)

    def test_launch_kwargs_hide_api(self):
        app = load_app()

        kwargs = app._launch_kwargs(share=True, auth=("u", "p"))

        self.assertEqual({"share": True, "auth": ("u", "p"), "show_api": False}, kwargs)

    def test_generate_uses_huggingface_token_helper(self):
        source = (ROOT / "generate.py").read_text()

        self.assertIn("def _get_hf_token", source)
        self.assertNotIn('open(os.path.expanduser("~/.cache/huggingface/token"))', source)

    def test_tokenizer_loads_without_remote_code_trust(self):
        app_source = (ROOT / "app.py").read_text()
        generate_source = (ROOT / "generate.py").read_text()

        self.assertNotIn("trust_remote_code=True", app_source)
        self.assertNotIn("trust_remote_code=True", generate_source)

    def test_serve_script_implements_documented_ngrok_tunnel(self):
        source = (ROOT / "serve.sh").read_text()

        self.assertIn("--tunnel", source)
        self.assertIn("NGROK_DOMAIN", source)
        self.assertIn("ngrok http", source)
        self.assertIn("tunnel.log", source)

    def test_public_copy_matches_public_caps_and_tone(self):
        site = (ROOT / "site/index.html").read_text()
        docs = (ROOT / "docs/public-demo.md").read_text()

        self.assertNotIn("12 steps max", site)
        self.assertNotIn("512 px max", site)
        self.assertNotIn("10 public queue slots", site)
        self.assertNotIn("bad time", site)
        self.assertNotIn("Dependency help", site)
        self.assertNotIn("receipts and install notes still stand", site)
        self.assertIn("20 steps max", site)
        self.assertIn("1024 px max", site)
        self.assertIn("one public queue slot", site)
        self.assertIn("Install notes", site)
        self.assertIn("If the Mac is asleep, or the demo is down, it might be over.", site)
        self.assertIn("--public --tunnel ngrok", docs)

    def test_readme_does_not_carry_provenance_claim(self):
        readme = (ROOT / "README.md").read_text()

        blocked = [
            "One " + "session",
            "Core build " + "burst",
            "## How it was " + "built",
            "Built from " + "scratch",
        ]
        for phrase in blocked:
            self.assertNotIn(phrase, readme)


if __name__ == "__main__":
    unittest.main()
