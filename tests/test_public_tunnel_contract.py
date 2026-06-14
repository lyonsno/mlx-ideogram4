import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SERVE_SH = ROOT / "serve.sh"
ROOT_INDEX = ROOT / "index.html"
LANDING = ROOT / "site" / "index.html"
LANDING_CONFIG = ROOT / "site" / "demo-config.js"
PUBLIC_DOC = ROOT / "docs" / "public-demo.md"


class PublicTunnelContractTest(unittest.TestCase):
    def test_serve_script_supports_stable_ngrok_tunnel(self):
        source = SERVE_SH.read_text()
        self.assertIn("--tunnel", source)
        self.assertIn("ngrok", source)
        self.assertIn("NGROK_DOMAIN", source)
        self.assertIn('ngrok http "$PORT" --url', source)
        self.assertIn("GRADIO_SERVER_PORT", source)
        self.assertIn("--tunnel-only", source)
        self.assertIn("TUNNEL_ONLY", source)

    def test_serve_script_uses_local_venv_not_uv_sync_for_live_server(self):
        source = SERVE_SH.read_text()
        self.assertIn(".venv/bin/python", source)
        self.assertNotIn("exec uv run", source)

    def test_static_landing_page_has_configured_demo_url(self):
        html = LANDING.read_text()
        config = LANDING_CONFIG.read_text()
        self.assertIn("demo-config.js", html)
        self.assertIn("NF4_DEMO_URL", config)
        self.assertIn("chummy-endless-caucus.ngrok-free.dev", config)
        self.assertIn("window.NF4_DEMO_URL", html)
        self.assertNotIn("gradio.live", html)

    def test_root_index_forwards_to_static_landing_page(self):
        source = ROOT_INDEX.read_text()
        self.assertIn("site/", source)
        self.assertIn("Ideogram4 NF4 on Apple Silicon", source)

    def test_reddit_landing_page_contract(self):
        source = LANDING.read_text()
        for text in [
            "Open live Mac demo",
            "View code",
            "official bitsandbytes NF4 weights",
            "MLX custom NF4 Metal kernels",
            "not MFLUX",
            "not PyTorch/MPS",
            "not a remote GPU",
            "512 px max",
            "20 steps max",
            "one active generation",
            "install the NF4 fork last",
            "Dependency help",
            "Ideogram 4 Non-Commercial License",
            "20260613T165202_nf4-mlx-metal_512x512_seed42.png",
            "2176225a6aeef6c042baf03a047014cb92e067ed45aadc78ceaae5492c70ec39",
        ]:
            self.assertIn(text, source)
        self.assertNotIn("YOUR-NGROK-DEV-DOMAIN", source)

    def test_public_demo_doc_names_no_paid_domain_path(self):
        source = PUBLIC_DOC.read_text()
        self.assertIn("ngrok-free.dev", source)
        self.assertIn("NGROK_DOMAIN", source)
        self.assertIn("Cloudflare", source)
        self.assertIn("random", source)


if __name__ == "__main__":
    unittest.main()
