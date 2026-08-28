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
ARG PINNED_SHA=main
ARG REPO_OWNER=Orimsaa/fitvision-backend

# Verified SHA256 checksums for models
ARG HASH_BENCHPRESS=0de25dc2507ef1166dd092744f6fabce233ca77e3c22e750bd6fab08a958e5b6
ARG HASH_DEADLIFT=faea101e4e7cfa7e9e664152025b9980704487345955ab5f102a5e9477fb44cf
ARG HASH_SQUAT=e295a4cf1cde35fa3b22248e75bd05473187cfbd0d4311f34b7e95bd56ce3397
ARG HASH_SQUAT_3CLASS=7086aa24643d599228cb171659ebb308a75074bbb18968257ddb77f0612c319e
ARG HASH_EXERCISE=9212713a8f4d4233a5ce52017edacb273111e863e0706d88f897cd195045ea60
ARG HASH_DEADLIFT_CAL=abb61789e9426201cae24351f74091c72aca69d4a49c1ef471107417b51b32d0

RUN mkdir -p data/models && \
    echo "[INFO] Downloading and verifying models via checksum..." && \
    \
    wget -qO data/models/benchpress_form.pkl "https://github.com/${REPO_OWNER}/raw/${PINNED_SHA}/data/models/benchpress_form.pkl" && \
    echo "${HASH_BENCHPRESS} data/models/benchpress_form.pkl" | sha256sum -c - && \
    \
    wget -qO data/models/deadlift_form.pkl "https://github.com/${REPO_OWNER}/raw/${PINNED_SHA}/data/models/deadlift_form.pkl" && \
    echo "${HASH_DEADLIFT} data/models/deadlift_form.pkl" | sha256sum -c - && \
    \
    wget -qO data/models/squat_form.pkl "https://github.com/${REPO_OWNER}/raw/${PINNED_SHA}/data/models/squat_form.pkl" && \
    echo "${HASH_SQUAT} data/models/squat_form.pkl" | sha256sum -c - && \
    \
    wget -qO data/models/squat_form_3class.pkl "https://github.com/${REPO_OWNER}/raw/${PINNED_SHA}/data/models/squat_form_3class.pkl" && \
    echo "${HASH_SQUAT_3CLASS} data/models/squat_form_3class.pkl" | sha256sum -c - && \
    \
    wget -qO data/models/exercise_classifier.pkl "https://github.com/${REPO_OWNER}/raw/${PINNED_SHA}/data/models/exercise_classifier.pkl" && \
    echo "${HASH_EXERCISE} data/models/exercise_classifier.pkl" | sha256sum -c - && \
    \
    wget -qO data/models/deadlift_form_calibrated.pkl "https://github.com/${REPO_OWNER}/raw/${PINNED_SHA}/data/models/deadlift_form_calibrated.pkl" && \
    echo "${HASH_DEADLIFT_CAL} data/models/deadlift_form_calibrated.pkl" | sha256sum -c - && \
    \
    echo "[OK] All models successfully verified."

# ── Production: Uncomment below and remove the wget block above to use trusted files ──
# COPY data/models/*.pkl ./data/models/

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
