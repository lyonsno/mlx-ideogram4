import ast
import json
import os
import pathlib
import unittest


APP_PATH = pathlib.Path(__file__).resolve().parents[1] / "app.py"


def load_public_helpers():
    tree = ast.parse(APP_PATH.read_text())
    keep = []
    wanted_functions = {
        "_build_prompt_payload",
        "_clamp_public_dimension",
        "_clamp_public_steps",
        "_effective_request",
        "_gallery_console_visibility_css",
        "_gallery_console_visibility_head",
        "_is_public_mode",
        "_queue_age_label",
        "_public_queue_admit",
        "_public_queue_is_full",
        "_public_queue_mark_finished",
        "_public_queue_mark_started",
        "_public_queue_admin_html",
        "_public_queue_snapshot",
        "_public_queue_status_html",
    }
    wanted_assigns = {
        "GALLERY_ADMIN_QUERY_PARAM",
        "GALLERY_ADMIN_QUERY_VALUE",
        "GALLERY_CONSOLE_ELEM_ID",
        "QUEUE_STATUS_ELEM_ID",
        "QUEUE_STATUS_TIMER_SECONDS",
        "QUEUE_ADMIN_ELEM_ID",
        "QUEUE_ADMIN_TIMER_SECONDS",
        "MODEL_REVISION",
        "PUBLIC_MAX_WIDTH",
        "PUBLIC_MAX_HEIGHT",
        "PUBLIC_MAX_STEPS",
        "PUBLIC_MIN_WIDTH",
        "PUBLIC_MIN_HEIGHT",
        "PUBLIC_MIN_STEPS",
        "PUBLIC_WIDTH",
        "PUBLIC_HEIGHT",
        "PUBLIC_PRESET",
            "PUBLIC_QUEUE_SIZE",
            "PUBLIC_STEPS",
            "PUBLIC_EXPECTED_LATENCY",
            "PUBLIC_GALLERY_FILES",
            "PUBLIC_STATUS_HEARTBEAT_SECONDS",
            "ROUTE_ID",
            "_queue_active",
            "_queue_lock",
            "_queue_next_id",
            "_queue_records",
            "_queue_tokens",
            "_queue_waiting",
        }
    for node in tree.body:
        if isinstance(node, ast.Import) and any(alias.name == "json" for alias in node.names):
            keep.append(node)
        if isinstance(node, ast.Import) and any(alias.name == "html" for alias in node.names):
            keep.append(node)
        if isinstance(node, ast.Import) and any(alias.name == "os" for alias in node.names):
            keep.append(node)
        if isinstance(node, ast.Import) and any(alias.name == "threading" for alias in node.names):
            keep.append(node)
        if isinstance(node, ast.Import) and any(alias.name == "time" for alias in node.names):
            keep.append(node)
        elif isinstance(node, ast.Assign):
            names = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if names & wanted_assigns:
                keep.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted_functions:
            keep.append(node)
    module = ast.Module(body=keep, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(APP_PATH), "exec"), namespace)
    return namespace


class PublicModeContractTest(unittest.TestCase):
    def test_public_mode_forces_launch_shape(self):
        ns = load_public_helpers()
        old = os.environ.get("NF4_PUBLIC_MODE")
        os.environ["NF4_PUBLIC_MODE"] = "1"
        try:
            prompt, use_json, preset, width, height, steps = ns["_effective_request"](
                "  a red cat on a blue couch  ",
                True,
                "V4_QUALITY_48",
                1024,
                1024,
                48,
            )
        finally:
            if old is None:
                os.environ.pop("NF4_PUBLIC_MODE", None)
            else:
                os.environ["NF4_PUBLIC_MODE"] = old

        self.assertEqual(prompt, "a red cat on a blue couch")
        self.assertFalse(use_json)
        self.assertEqual(preset, "V4_DEFAULT_20")
        self.assertEqual(width, 512)
        self.assertEqual(height, 512)
        self.assertEqual(steps, 20)

    def test_public_mode_allows_smaller_launch_shape(self):
        ns = load_public_helpers()
        old = os.environ.get("NF4_PUBLIC_MODE")
        os.environ["NF4_PUBLIC_MODE"] = "1"
        try:
            prompt, use_json, preset, width, height, steps = ns["_effective_request"](
                "small square",
                False,
                "V4_DEFAULT_20",
                384,
                320,
                12,
            )
        finally:
            if old is None:
                os.environ.pop("NF4_PUBLIC_MODE", None)
            else:
                os.environ["NF4_PUBLIC_MODE"] = old

        self.assertEqual((width, height, steps), (384, 320, 12))

    def test_public_mode_contract_constants_name_the_demo(self):
        ns = load_public_helpers()
        self.assertEqual(ns["PUBLIC_WIDTH"], 512)
        self.assertEqual(ns["PUBLIC_HEIGHT"], 512)
        self.assertEqual(ns["PUBLIC_MAX_WIDTH"], 512)
        self.assertEqual(ns["PUBLIC_MAX_HEIGHT"], 512)
        self.assertEqual(ns["PUBLIC_MIN_WIDTH"], 256)
        self.assertEqual(ns["PUBLIC_MIN_HEIGHT"], 256)
        self.assertEqual(ns["PUBLIC_STEPS"], 20)
        self.assertEqual(ns["PUBLIC_MAX_STEPS"], 20)
        self.assertEqual(ns["PUBLIC_MIN_STEPS"], 4)
        self.assertEqual(ns["PUBLIC_PRESET"], "V4_DEFAULT_20")
        self.assertEqual(ns["PUBLIC_QUEUE_SIZE"], 10)
        self.assertIn("10", ns["PUBLIC_EXPECTED_LATENCY"])

    def test_prompt_builder_outputs_known_good_shape(self):
        ns = load_public_helpers()
        payload = ns["_build_prompt_payload"](
            "The bold letters NF4 in black",
            True,
            "minimal graphic design",
            "flat",
            "digital vector art",
            "#000000, #FFFFFF",
            True,
            "512 x 512 square canvas",
            "pure white",
            "centered single element",
            "text | Large bold black letters NF4\nobj | Apple logo silhouette",
        )
        data = json.loads(payload)
        self.assertEqual(data["high_level_description"], "The bold letters NF4 in black")
        self.assertEqual(data["style_description"]["color_palette"], ["#000000", "#FFFFFF"])
        self.assertEqual(data["compositional_deconstruction"]["layout"], "centered single element")
        self.assertEqual(data["compositional_deconstruction"]["elements"][0]["type"], "text")

    def test_public_route_identity_and_gallery_contract(self):
        ns = load_public_helpers()
        self.assertEqual(ns["ROUTE_ID"], "nf4-mlx-metal")
        self.assertEqual(ns["MODEL_REVISION"], "f664347839e0a87bc495f5c9483cc0014b8e344e")
        self.assertIn("evidence/smoke_16gb_512_20step.png", ns["PUBLIC_GALLERY_FILES"])
        self.assertIn("evidence/hero_nf4_apple_1024x1024.png", ns["PUBLIC_GALLERY_FILES"])

    def test_waiting_gallery_has_rotating_preview_contract(self):
        source = APP_PATH.read_text()
        self.assertIn("GALLERY_ROTATE_SECONDS", source)
        self.assertIn("_featured_wait_output_image", source)
        self.assertNotIn('gr.Markdown("### While You Wait")', source)
        self.assertNotIn("wait_preview = gr.Image", source)
        self.assertNotIn("outputs=[wait_preview, carousel_index]", source)

    def test_featured_gallery_requires_explicit_promotion(self):
        source = APP_PATH.read_text()
        self.assertIn('promoted = list(dict.fromkeys(data.get("promoted", [])))', source)
        self.assertIn('"promoted": promoted', source)
        self.assertIn('"shown": shown', source)
        self.assertIn("if promoted:\n        return promoted\n    return []", source)
        self.assertIn("rel not in promoted and rel not in shown and rel not in hidden", source)
        self.assertIn("rel not in promoted and rel not in demoted", source)

    def test_public_queue_allows_multiple_waiting_jobs(self):
        source = APP_PATH.read_text()
        self.assertIn("demo.queue(max_size=PUBLIC_QUEUE_SIZE, default_concurrency_limit=1)", source)
        self.assertIn("public queue holds up to {PUBLIC_QUEUE_SIZE} jobs", source)

    def test_public_queue_admission_counts_and_full_state(self):
        ns = load_public_helpers()
        ns["_queue_active"] = 0
        ns["_queue_waiting"] = 0
        ns["_queue_next_id"] = 0
        ns["_queue_tokens"] = {}
        ns["_queue_records"] = {}

        tokens = []
        for _ in range(ns["PUBLIC_QUEUE_SIZE"]):
            token, message = ns["_public_queue_admit"]()
            self.assertTrue(token)
            self.assertIn("Queue slot reserved", message)
            tokens.append(token)

        snapshot = ns["_public_queue_snapshot"]()
        self.assertEqual(snapshot["active"], 0)
        self.assertEqual(snapshot["waiting"], ns["PUBLIC_QUEUE_SIZE"])
        self.assertEqual(snapshot["free"], 0)
        self.assertTrue(snapshot["full"])
        self.assertEqual(len(snapshot["entries"]), ns["PUBLIC_QUEUE_SIZE"])
        self.assertEqual(snapshot["entries"][0]["token"], tokens[0])
        self.assertEqual(snapshot["entries"][0]["state"], "waiting")
        self.assertEqual(snapshot["entries"][0]["position"], 1)
        self.assertTrue(ns["_public_queue_is_full"]())

        token, message = ns["_public_queue_admit"]()
        self.assertIsNone(token)
        self.assertIn("Queue full", message)

        self.assertTrue(ns["_public_queue_mark_started"](tokens[0]))
        snapshot = ns["_public_queue_snapshot"]()
        self.assertEqual(snapshot["active"], 1)
        self.assertEqual(snapshot["waiting"], ns["PUBLIC_QUEUE_SIZE"] - 1)
        self.assertEqual(snapshot["entries"][0]["state"], "active")
        self.assertIsNotNone(snapshot["entries"][0]["started_at"])
        self.assertTrue(snapshot["full"])

        ns["_public_queue_mark_finished"](tokens[0])
        snapshot = ns["_public_queue_snapshot"]()
        self.assertEqual(snapshot["active"], 0)
        self.assertEqual(snapshot["waiting"], ns["PUBLIC_QUEUE_SIZE"] - 1)
        self.assertEqual(snapshot["free"], 1)
        self.assertFalse(snapshot["full"])
        self.assertNotIn(tokens[0], [entry["token"] for entry in snapshot["entries"]])

    def test_public_queue_admin_html_lists_entries(self):
        ns = load_public_helpers()
        ns["_queue_active"] = 0
        ns["_queue_waiting"] = 0
        ns["_queue_next_id"] = 0
        ns["_queue_tokens"] = {}
        ns["_queue_records"] = {}

        first, _ = ns["_public_queue_admit"]()
        second, _ = ns["_public_queue_admit"]()
        self.assertTrue(ns["_public_queue_mark_started"](first))

        html = ns["_public_queue_admin_html"]()
        self.assertIn(ns["QUEUE_ADMIN_ELEM_ID"], html)
        self.assertIn("Admin queue", html)
        self.assertIn("1 active", html)
        self.assertIn("1 waiting", html)
        self.assertIn(first, html)
        self.assertIn(second, html)
        self.assertIn("active", html)
        self.assertIn("waiting", html)
        self.assertIn("Admitted", html)
        self.assertIn("Started", html)
        self.assertIn("Refreshed every", html)

    def test_public_queue_admission_message_counts_active_job_ahead(self):
        ns = load_public_helpers()
        ns["_queue_active"] = 0
        ns["_queue_waiting"] = 0
        ns["_queue_next_id"] = 0
        ns["_queue_tokens"] = {}
        ns["_queue_records"] = {}

        active_token, _ = ns["_public_queue_admit"]()
        self.assertTrue(ns["_public_queue_mark_started"](active_token))
        next_token, message = ns["_public_queue_admit"]()

        self.assertTrue(next_token)
        self.assertIn("Queue slot reserved", message)
        self.assertIn("waiting behind 1 public job", message)

    def test_public_queue_status_strip_and_button_wiring(self):
        ns = load_public_helpers()
        ns["_queue_active"] = 1
        ns["_queue_waiting"] = 3
        ns["_queue_next_id"] = 4
        ns["_queue_tokens"] = {}
        ns["_queue_records"] = {}
        html = ns["_public_queue_status_html"]()
        self.assertIn(ns["QUEUE_STATUS_ELEM_ID"], html)
        self.assertIn("Queue:", html)
        self.assertIn("1 running", html)
        self.assertIn("3 waiting", html)
        self.assertIn("6 free", html)
        self.assertEqual(ns["QUEUE_STATUS_TIMER_SECONDS"], 3)
        self.assertEqual(ns["QUEUE_ADMIN_TIMER_SECONDS"], 3)

        source = APP_PATH.read_text()
        self.assertIn("admission_token = gr.State", source)
        self.assertIn("queue_status = gr.HTML", source)
        self.assertIn("queue_timer = gr.Timer", source)
        self.assertIn("queue_timer.tick", source)
        self.assertIn("queue_admin = gr.HTML", source)
        self.assertIn("queue_admin_timer = gr.Timer", source)
        self.assertIn("queue_admin_timer.tick", source)
        self.assertIn("_admit_generate_click", source)
        self.assertIn("_generate_with_admission", source)
        self.assertIn("admit_event.success", source)
        self.assertIn("gr.update(interactive=not _public_queue_is_full())", source)

    def test_public_queue_hydrates_status_button_and_admin_on_page_load(self):
        source = APP_PATH.read_text()
        self.assertIn("def _queue_initial_controls", source)
        self.assertIn("demo.load(", source)
        self.assertIn("fn=_queue_initial_controls", source)
        self.assertIn("outputs=[queue_status, btn, queue_admin]", source)
        self.assertLess(
            source.index("demo.load("),
            source.index("admit_event = btn.click("),
        )

    def test_public_generation_has_elapsed_heartbeat(self):
        ns = load_public_helpers()
        source = APP_PATH.read_text()
        self.assertEqual(ns["PUBLIC_STATUS_HEARTBEAT_SECONDS"], 5)
        self.assertIn("selectors.DefaultSelector()", source)
        self.assertIn("selector.select(timeout=timeout)", source)
        self.assertIn("Waiting for generate.py output...", source)
        self.assertIn("Elapsed: {elapsed:.0f}s", source)

    def test_gallery_visibility_console_contract(self):
        source = APP_PATH.read_text()
        self.assertIn('"hidden"', source)
        self.assertIn("_visible_generated_images", source)
        self.assertIn("Gallery visibility console", source)
        self.assertIn('gr.Markdown("#### Showing")', source)
        self.assertIn('gr.Markdown("#### Hidden")', source)
        self.assertIn("Hide from Gallery", source)
        self.assertIn("Show in Gallery", source)
        self.assertIn("GALLERY_SLOT_COUNT = 40", source)
        self.assertIn("visible_slot_tiles = []", source)
        self.assertIn("hidden_slot_tiles = []", source)
        self.assertIn("_gallery_slot_html", source)
        self.assertIn("gallery-slot", source)
        self.assertIn("selected_class", source)
        self.assertNotIn("selected_preview = gr.Image", source)
        self.assertNotIn("visible_slot_buttons = []", source)
        self.assertNotIn("hidden_slot_buttons = []", source)
        self.assertNotIn('gr.Button(\n                                "Select"', source)
        self.assertIn("_select_visible_slot", source)
        self.assertIn("_select_hidden_slot", source)
        self.assertIn("_gallery_slot_updates", source)
        self.assertIn("_hide_gallery_item", source)
        self.assertIn("_show_gallery_item", source)

    def test_public_gallery_console_is_operator_query_gated(self):
        ns = load_public_helpers()
        self.assertEqual(ns["GALLERY_CONSOLE_ELEM_ID"], "gallery-visibility-console")
        self.assertEqual(ns["GALLERY_ADMIN_QUERY_PARAM"], "gallery_admin")
        self.assertEqual(ns["GALLERY_ADMIN_QUERY_VALUE"], "1")

        public_css = ns["_gallery_console_visibility_css"](True)
        self.assertIn("#gallery-visibility-console", public_css)
        self.assertIn("display: none", public_css)
        self.assertIn(".gallery-admin-enabled #gallery-visibility-console", public_css)
        self.assertIn("html.gallery-admin-enabled .gradio-container .contain #gallery-visibility-console", public_css)

        local_css = ns["_gallery_console_visibility_css"](False)
        self.assertNotIn("display: none", local_css)

        head = ns["_gallery_console_visibility_head"]()
        self.assertIn("<script>", head)
        self.assertIn("URLSearchParams(window.location.search)", head)
        self.assertIn("gallery_admin", head)
        self.assertIn("gallery-admin-enabled", head)
        self.assertIn("DOMContentLoaded", head)

        source = APP_PATH.read_text()
        self.assertIn('elem_id=GALLERY_CONSOLE_ELEM_ID', source)
        self.assertIn('open=_PUBLIC_MODE, elem_id=GALLERY_CONSOLE_ELEM_ID', source)
        self.assertIn("head=_gallery_console_visibility_head()", source)

    def test_prompt_presets_and_randomizer_contract(self):
        source = APP_PATH.read_text()
        for name in [
            "PROMPT_PRESETS",
            "AESTHETICS_PRESETS",
            "LIGHTING_PRESETS",
            "MEDIUM_PRESETS",
            "COLOR_PALETTE_PRESETS",
            "CANVAS_PRESETS",
            "BACKGROUND_PRESETS",
            "LAYOUT_PRESETS",
            "ELEMENTS_PRESETS",
            "ADVANCED_RUN_PRESETS",
        ]:
            self.assertIn(name, source)
        self.assertIn("Randomize All", source)
        self.assertIn("_randomize_form", source)
        self.assertIn("_apply_advanced_preset", source)
        self.assertIn("_preset_choices", source)


if __name__ == "__main__":
    unittest.main()
