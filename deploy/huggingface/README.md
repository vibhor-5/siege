---
title: SIEGE — Interpretability Arena (OpenEnv)
emoji: 🔴
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
license: bsd-3-clause
pinned: false
---

# SIEGE on Hugging Face Spaces

OpenEnv-compatible environment: **Red vs Blue** in a real LM forward pass, served as FastAPI + optional **OpenEnv web UI** (`ENABLE_WEB_INTERFACE=true` in the Docker image).

## After this Space builds

- Open the **default URL** of the Space — if the web UI is enabled, you get the OpenEnv interface.
- Or call the HTTP API (same routes your local `uvicorn` used): e.g. health and env endpoints under the root URL.

**Tip:** First load may be slow while `transformer-lens` downloads `MODEL_NAME` (default `Qwen/Qwen2.5-0.5B-Instruct`). If the Hub requests a license acceptance, set a `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) in **Settings → Repository secrets** for this Space.

## Hardware

- **CPU (free tier):** works for the 0.5B run; can be slow on first download.
- **GPU (optional):** upgrade the Space to a small GPU and set `DEVICE=cuda` in **Settings → Repository variables** if you wire CUDA in a custom image later (this Dockerfile is CPU-only).

## Env vars (optional)

| Variable | Default | Purpose |
|----------|---------|--------|
| `MODEL_NAME` | `Qwen/Qwen2.5-0.5B-Instruct` | Target LM in the arena |
| `DEVICE` | `cpu` | `cpu` only in this image |
| `SAFETY_MODE` | `keyword` | … |
| `ENABLE_WEB_INTERFACE` | `true` in HF Dockerfile | Set `false` for API-only |
| `PORT` | `7860` | Set by Spaces; do not override unless you know the proxy port |

## Source

See the upstream project repo for full docs, `train.py` / `train_grpo`, and the OpenEnv `client.py` wire format.
