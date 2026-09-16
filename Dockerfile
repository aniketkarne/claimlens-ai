# ClaimLens AI — backend Dockerfile
#
# Local-first FastAPI service for European motor accident claim processing.
# This image installs the Python deps + Tesseract OCR. Ollama and the
# `qwen2.5:7b` / `qwen3-vl:4b` models are expected to run on the host
# (see README "Setup Instructions") and are reached over HTTP at
# `http://host.docker.internal:11434` from the container.
#
# Build:
#   docker build -t claimlens-api .
#
# Run (Ollama on host):
#   docker run --rm -p 8000:8000 \
#     -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
#     -v "$PWD/logs:/app/logs" \
#     -v "$PWD/uploads:/app/uploads" \
#     claimlens-api

FROM python:3.11-slim-bookworm

# System deps:
# - tesseract-ocr + tesseract-ocr-nld: OCR for Dutch + English claim forms
# - poppler-utils: required by pdf2image for PDF -> image conversion
# - libgl1 / libglib2.0-0: opencv-python-headless runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-nld \
    tesseract-ocr-eng \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so Docker can cache the layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source.
COPY app/ ./app/

# Persist logs and uploads across container restarts via volume mounts
# (see docker run example above).
RUN mkdir -p /app/logs /app/uploads

EXPOSE 8000

# Use the FastAPI entry point.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]