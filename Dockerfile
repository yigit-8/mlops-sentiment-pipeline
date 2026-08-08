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

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
