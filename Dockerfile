# Internal ML service. Build from the repository root:
#   docker build -t mlengine-service .
#   docker run -p 8100:8100 -e ML_SERVICE_TOKEN=change-me mlengine-service
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
COPY mlengine ./mlengine
RUN pip install --no-cache-dir ".[service]"
COPY models ./models
ENV MODEL_REGISTRY_PATH=/app/models
EXPOSE 8100
CMD ["uvicorn", "mlengine.service.app:app", "--host", "0.0.0.0", "--port", "8100"]
