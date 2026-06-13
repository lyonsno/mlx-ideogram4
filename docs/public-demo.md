# Public Demo Front Door

The durable public path should be:

1. A static landing page from `site/`.
2. A stable tunnel URL from a free ngrok dev domain.
3. The Mac-local Gradio server behind that tunnel.

Do not put a raw `*.gradio.live` URL in a Reddit post. Gradio's share links are useful for quick smokes, but they are temporary tunnel URLs. A post should link to a stable front door, and that page should point at the current live demo endpoint.

## No-paid-domain tunnel

Use ngrok's free account dev domain. It is an assigned `ngrok-free.dev` hostname that stays with the account, so it solves the "same link after restart" problem without buying a domain.

One-time setup:

```sh
brew install ngrok/ngrok/ngrok
ngrok config add-authtoken "$NGROK_AUTHTOKEN"
```

Find the assigned domain in the ngrok dashboard under `Universal Gateway > Domains`. Then launch:

```sh
NGROK_DOMAIN=your-assigned-name.ngrok-free.dev ./serve.sh --public --tunnel ngrok
```

The script starts the local Gradio app on `GRADIO_SERVER_PORT` or port `7860`, waits for it to answer locally, then starts:

```sh
ngrok http "${GRADIO_SERVER_PORT:-7860}" --url "https://$NGROK_DOMAIN"
```

Logs go to:

- `~/Library/Logs/ideogram4-nf4/serve.log`
- `~/Library/Logs/ideogram4-nf4/tunnel.log`

## Landing page update

Set the live demo URL in `site/demo-config.js`:

```js
window.NF4_DEMO_URL = "https://your-assigned-name.ngrok-free.dev";
```

The page intentionally keeps the front door separate from the tunnel. If the Mac is offline, the page still explains the project and points to the repo; when the tunnel is up, the live demo button points at the Mac.

## Why not Cloudflare first?

Cloudflare Quick Tunnels are excellent for zero-account tests, but the no-domain quick tunnel path gives a random `trycloudflare.com` subdomain. A named persistent Cloudflare Tunnel is the stronger production shape, but it wants a Cloudflare-managed DNS zone or domain. For "stable URL, no paid domain, right now," ngrok's free dev domain is the cleaner first move.

## Why not localhost.run first?

The free localhost.run flow is pleasantly simple SSH, but its free tunnel names can change. Stable custom names are a paid/custom-domain path there, so it does not beat ngrok for this exact Reddit-link problem.
