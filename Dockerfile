# Hugging Face Spaces (sdk: docker) and generic container deploy.
# Listens on $PORT (7860 on HF). Build from repo root: docker build -t siege .
#
# For docs see deploy/huggingface/DEPLOY.md

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY server/requirements.txt /app/server/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/server/requirements.txt

COPY interp_arena/ /app/interp_arena/
COPY models.py client.py __init__.py /app/
COPY server/ /app/server/
COPY data/ /app/data/
COPY configs/ /app/configs/

ENV HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/hub

ENV MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct" \
    DEVICE="cpu" \
    SAFETY_MODE="keyword" \
    ENABLE_WEB_INTERFACE="true" \
    PORT=7860

EXPOSE 7860

CMD ["sh", "-c", "exec uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
