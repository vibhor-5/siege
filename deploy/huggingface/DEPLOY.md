# Publish SIEGE to Hugging Face Spaces

The container definition lives at the **repo root** as `Dockerfile` (for HF’s default “build from root” flow).

## Option A — One Space = this full GitHub repo (recommended)

1. On [Hugging Face new Space](https://huggingface.co/new-space), create a **Docker** Space and link the same remote as this project (or push a mirror to `https://huggingface.co/spaces/<user>/<name>` and add the GitHub remote).
2. **Settings → Configuration:** SDK **Docker**, use default `Dockerfile` at repository root. Build context = repo root.
3. **Secrets (optional):** set `HUGGING_FACE_HUB_TOKEN` if the configured `MODEL_NAME` is gated.
4. Wait for the build; open the Space URL. First boot may take a while while the model is downloaded.

## Option B — OpenEnv CLI (`openenv push`)

The repo is an [OpenEnv](https://meta-pytorch.org/OpenEnv) environment (see `openenv.yaml` at the repo root). With the [Hugging Face Hub CLI](https://huggingface.co/docs/huggingface_hub/quick-start) logged in (`huggingface-cli login` or `HF_TOKEN` in the environment):

```bash
cd /path/to/siege
uv run openenv validate                    # static checks
uv run openenv validate --url http://127.0.0.1:8000   # runtime: start server first
# Push a Docker Space (defaults to <your_hf_user>/interp-arena from openenv.yaml name, or set --repo-id)
uv run openenv push --repo-id YOUR_HF_USER/interp-arena
```

`openenv push` stages the repo, prefers the **root** `Dockerfile` when present, and can inject `ENABLE_WEB_INTERFACE=true` for the Gradio web UI. Ensure `/metadata` is fast: this project loads the target model only on `reset`/`step`, not on every metadata probe (required for `openenv validate --url`).

## Local check before pushing

```bash
cd /path/to/siege
docker build -t siege-hf .
docker run --rm -p 7860:7860 \
  -e HUGGING_FACE_HUB_TOKEN=optional_if_gated \
  siege-hf
# Visit http://localhost:7860
```

## Port

Hugging Face Spaces expect the app to listen on **`PORT`** (typically **7860**). The provided Dockerfile uses `uvicorn` with `host 0.0.0.0` and that port. Do not bind to `8000` on the Space.

## If the build runs out of RAM

Use a **smaller** `MODEL_NAME` or a **larger** Space (CPU + RAM) tier. The 0.5B model is already the default small target.
