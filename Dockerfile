# Stage 1: Build
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build essentials (needed by some packages like scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Copy only the slim requirements for deploy
COPY requirements-fly.txt .
RUN pip install --no-cache-dir --user -r requirements-fly.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application files
COPY app/ ./app/
COPY config/ ./config/
COPY src/ ./src/
COPY data/models/*.pkl ./data/models/

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:
ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
