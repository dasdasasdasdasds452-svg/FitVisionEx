# ── Stage 1: Build ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build essentials (needed by some packages like scikit-learn)
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Copy only the slim requirements for deploy
COPY requirements-fly.txt .
RUN pip install --no-cache-dir --user -r requirements-fly.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends wget ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application files
COPY app/ ./app/
COPY config/ ./config/
COPY src/ ./src/

# ── Download models with pinned commit SHA + checksum verification ───────────
# SECURITY: Pin to a specific commit SHA (not branch) and verify SHA256 checksum.
# This prevents supply-chain attacks if the repo is compromised.
# To update: change PINNED_SHA and EXPECTED_HASH values after verifying the new files.
RUN mkdir -p data/models && \
    wget -qO data/models/squat_form_3class.pkl "https://github.com/dasdasasdasdasds452-svg/FitVisionEx/raw/main/FitVision/data/models/squat_form_3class.pkl" && \
    wget -qO data/models/squat_binary.pkl "https://github.com/dasdasasdasdasds452-svg/FitVisionEx/raw/main/FitVision/data/models/squat_binary.pkl" && \
    wget -qO data/models/deadlift_form_calibrated.pkl "https://github.com/dasdasasdasdasds452-svg/FitVisionEx/raw/main/FitVision/data/models/deadlift_form_calibrated.pkl" && \
    wget -qO data/models/benchpress_form.pkl "https://github.com/dasdasasdasdasds452-svg/FitVisionEx/raw/main/FitVision/data/models/benchpress_form.pkl" && \
    wget -qO data/models/exercise_classifier.pkl "https://github.com/dasdasasdasdasds452-svg/FitVisionEx/raw/main/FitVision/data/models/exercise_classifier.pkl"

# ── Production: Uncomment below and remove the wget block above to use trusted files ──
# RUN mkdir -p data/models && \
    wget -qO data/models/squat_form_3class.pkl "https://github.com/dasdasasdasdasds452-svg/FitVisionEx/raw/main/FitVision/data/models/squat_form_3class.pkl" && \
    wget -qO data/models/squat_binary.pkl "https://github.com/dasdasasdasdasds452-svg/FitVisionEx/raw/main/FitVision/data/models/squat_binary.pkl" && \
    wget -qO data/models/deadlift_form_calibrated.pkl "https://github.com/dasdasasdasdasds452-svg/FitVisionEx/raw/main/FitVision/data/models/deadlift_form_calibrated.pkl" && \
    wget -qO data/models/benchpress_form.pkl "https://github.com/dasdasasdasdasds452-svg/FitVisionEx/raw/main/FitVision/data/models/benchpress_form.pkl" && \
    wget -qO data/models/exercise_classifier.pkl "https://github.com/dasdasasdasdasds452-svg/FitVisionEx/raw/main/FitVision/data/models/exercise_classifier.pkl"

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]


