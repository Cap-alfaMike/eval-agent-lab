FROM python:3.11-slim AS base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY src/ ./src/
COPY datasets/ ./datasets/

# Install package
RUN pip install --no-cache-dir -e .

# --- CLI Mode ---
FROM base AS cli
ENTRYPOINT ["eval-agent-lab"]
CMD ["--help"]

# --- API Mode ---
FROM base AS api
EXPOSE 8000
CMD ["uvicorn", "eval_agent_lab.api:app", "--host", "0.0.0.0", "--port", "8000"]
