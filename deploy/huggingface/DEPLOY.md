# Publish SIEGE to Hugging Face Spaces

The container definition lives at the **repo root** as `Dockerfile` (for HF’s default “build from root” flow).

## Option A — One Space = this full GitHub repo (recommended)

1. On [Hugging Face new Space](https://huggingface.co/new-space), create a **Docker** Space and link the same remote as this project (or push a mirror to `https://huggingface.co/spaces/<user>/<name>` and add the GitHub remote).
2. **Settings → Configuration:** SDK **Docker**, use default `Dockerfile` at repository root. Build context = repo root.
3. **Secrets (optional):** set `HUGGING_FACE_HUB_TOKEN` if the configured `MODEL_NAME` is gated.
4. Wait for the build; open the Space URL. First boot may take a while while the model is downloaded.

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
