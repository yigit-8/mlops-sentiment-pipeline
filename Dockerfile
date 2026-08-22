# Mutable tag on purpose: it picks up base-image security patches on rebuild.
# The tradeoff is that two builds of the same commit are not byte-identical; pin
# a digest (FROM python:3.11-slim@sha256:...) if reproducible builds matter more
# than automatic patching.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# The contents of data/ are generated at runtime and git-ignored, so the
# directory does not exist in a fresh checkout: create it instead of copying.
RUN mkdir -p data

# Cache inside /app so the non-root user below can still read it, and name the
# model explicitly: an unpinned pipeline() caches whatever Hugging Face ships as
# the default, which is not necessarily the DEFAULT_MODEL that serve.py loads.
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')"

RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# curl is not installed in this image, so probe with the interpreter itself.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0) if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status == 200 else sys.exit(1)"

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
