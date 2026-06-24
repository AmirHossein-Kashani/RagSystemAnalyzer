FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/app/.hf \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# Install CPU-only torch first so sentence-transformers picks it up instead of
# pulling the much-larger CUDA wheel from PyPI's default index.
RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu torch

COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download the embedding model so the running container needs no internet
# and the first request is fast.
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}', device='cpu')"

# Application code + default LLM config.
COPY app/ ./app/
COPY llm_config.json llm_config.openai.json ./

# Non-root user, data dirs seeded so a fresh volume mount inherits the layout.
RUN useradd --create-home --uid 10001 app \
 && mkdir -p /app/data/docs /app/data/chroma \
 && chown -R app:app /app

USER app

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request, sys; \
      sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8765/api/health', timeout=3).status == 200 else 1)" \
      || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8765"]
