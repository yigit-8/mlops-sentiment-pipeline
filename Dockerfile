FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# The contents of data/ are generated at runtime and git-ignored, so the
# directory does not exist in a fresh checkout: create it instead of copying.
RUN mkdir -p data

RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis')"

EXPOSE 8000

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
