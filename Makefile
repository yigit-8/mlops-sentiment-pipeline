.PHONY: install test lint run docker-up docker-down clean

install:
	pip install -r requirements.txt ruff black pytest mlflow evidently

lint:
	ruff check src tests --fix
	black src tests

test:
	pytest tests/ -v

run:
	uvicorn src.serve:app --reload

train:
	python src/train.py

drift:
	python src/drift.py

docker-up:
	docker compose up --build -d

docker-down:
	docker compose down

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ src/__pycache__ tests/__pycache__ mlruns
