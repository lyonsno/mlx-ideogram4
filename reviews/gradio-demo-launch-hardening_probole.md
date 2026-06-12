# Probolē: Gradio Demo Launch Hardening

## Target

- `/Users/noahlyons/dev/mlx-ideogram4/app.py` (Gradio UI)
- `/Users/noahlyons/dev/mlx-ideogram4/serve.sh` (LaunchAgent wrapper)
- `/Users/noahlyons/dev/mlx-ideogram4/generate.py` (CLI + NF4 guard)
- `/Users/noahlyons/dev/mlx-ideogram4/README.md` (public-facing text)

## Review scope

Launch readiness review for a public-facing Gradio demo that will be linked
from Reddit. This is running on a real Mac serving real GPU compute to
strangers. Review for: security, resource exhaustion, misleading claims,
broken install paths, and UX footguns.

## Review context mode

Target files only. No inherited implementation context.

## What to look for

### Security / resource exhaustion
1. Rate limiting: is the 30s cooldown sufficient? Can it be bypassed?
2. Queue: max_size=1, but can someone hold the queue indefinitely with a slow prompt?
3. Resolution clamp: does it actually prevent >1024 inputs from the slider or API?
4. Memory: with previews disabled in public mode, is 16 GB safe? Any OOM paths?
5. Prompt injection: can a malicious prompt crash the server or consume unbounded memory?
6. The LaunchAgent plist: is it safe to have `--share --public` auto-start on login?

### Public-facing claims
7. README performance numbers: do they match the receipts? Any overclaiming?
8. "Model weights under non-commercial license" — is this visible enough?
9. Memory definition clarity: "MLX-reported peak active during sampling"
10. The "any bitsandbytes NF4 model" language — sufficiently caveated?

### Install path
11. Can a stranger actually install from the README six-command flow?
12. Does the `--force-reinstall --no-deps` trick survive `pip install -e .` pulling stock MLX?
13. Does `_assert_nf4_available()` catch all failure modes?
14. Is the `uv run` alternative path actually tested?

### UX
15. Gradio antechamber: is the NF4 explainer accurate?
16. Gallery: does it load images correctly from evidence/?
17. Does the status text update properly without previews?
18. Can someone click Generate twice and crash things?

## Out of scope

- NF4 kernel correctness (reviewed separately)
- Model architecture parity (reviewed separately)
- Upstream MLX PR readiness
