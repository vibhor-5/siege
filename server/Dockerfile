FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY interp_arena/ ./interp_arena/
COPY models.py client.py __init__.py ./
COPY server/ ./server/
COPY data/ ./data/
COPY configs/ ./configs/

ENV MODEL_NAME="Qwen/Qwen2.5-0.5B-Instruct"
ENV DEVICE="cpu"
ENV SAFETY_MODE="keyword"
ENV ENABLE_WEB_INTERFACE="false"

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
