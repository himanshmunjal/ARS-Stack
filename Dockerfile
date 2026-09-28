# ARS-Stack reproducible environment.
# Builds one image used for the full pipeline, the Prefect flow, and the
# FastAPI inference endpoint — see docker-compose.yml for how each service
# invokes it differently.

FROM python:3.11-slim

WORKDIR /app

# System deps needed to build a couple of the ML wheels (lightgbm/xgboost)
# on slim images that lack them by default.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

EXPOSE 8000

# Default: serve the trained models. Override the command for pipeline/DAG runs, e.g.:
#   docker compose run pipeline python -m src.pipeline --dataset creditcard
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
