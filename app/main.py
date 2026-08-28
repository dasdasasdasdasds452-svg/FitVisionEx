"""
FitVision — FastAPI main app
Run: uvicorn app.main:app --reload --port 8000
"""
import os
import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from contextlib import asynccontextmanager

sys.path.append(str(Path(__file__).parent.parent))

from app.schemas import Features13, SquatFeatures, ExercisePrediction, FormPrediction
from app.predictor import predict_exercise, predict_deadlift, predict_squat, predict_benchpress

import structlog
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

sentry_dsn = os.getenv("SENTRY_DSN")
if sentry_dsn:
    sentry_sdk.init(
        dsn=sentry_dsn,
        traces_sample_rate=1.0,
        profiles_sample_rate=1.0,
        integrations=[FastApiIntegration()]
    )

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer() if os.getenv("JSON_LOGS") else structlog.dev.ConsoleRenderer()
    ]
)
logger = structlog.get_logger("fitvision")
# ── CORS ──────────────────────────────────────────────────────────────────────
# Configure via env var ALLOWED_ORIGINS="https://fitvision.app,https://www.fitvision.app"
# Default to "*" only for local dev — set explicit origins in production!
_raw_origins = os.getenv("ALLOWED_ORIGINS", "").strip()
_allowed_origins = (
    [o.strip() for o in _raw_origins.split(",") if o.strip()]
    if _raw_origins
    else ["*"]
)

# ── Health (cache model list at startup to avoid repeated disk I/O) ───────────
_cached_health_models: list[str] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    logger.info("FitVision API Starting up. Models will be lazy-loaded on demand to save memory.")
    global _cached_health_models
    models_dir = Path(__file__).parent.parent / "data" / "models"
    _cached_health_models = (
        [f.name for f in models_dir.glob("*.pkl")]
        if models_dir.exists()
        else []
    )
    logger.info("Ready!")
    yield
    # shutdown
    logger.info("Shutting down...")

app = FastAPI(title="FitVision API", version="2.0", lifespan=lifespan)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error('Unhandled exception', error=str(exc), path=request.url.path)
    return JSONResponse(status_code=500, content={'detail': 'Internal server error'})

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type"],
)

# ── Rate Limiting (slowapi) ───────────────────────────────────────────────────
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

limiter = Limiter(key_func=get_remote_address, default_limits=["30/minute"])
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Please try again later."},
    )

# ── Serve static web files ────────────────────────────────────────────────────
WEB_DIR = Path(__file__).parent.parent / "web"
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

@app.get("/")
async def index():
    index_path = WEB_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"status": "FitVision API running", "docs": "/docs"}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models": _cached_health_models,
    }


# ── Predict Endpoints (def = FastAPI runs in threadpool, NOT async blocking) ──
from pydantic import BaseModel
from typing import Any, Optional

class ReportPayload(BaseModel):
    exercise: str
    features: dict[str, Any] | list[float]
    predicted_correct: bool
    actual_correct: bool
    user_feedback: Optional[str] = None

@app.post("/report")
@limiter.limit("10/minute")
def report_prediction(request: Request, body: ReportPayload):
    """
    Data Flywheel API: Collect incorrect predictions to retrain models.
    Saves reported features and correct labels to a CSV/JSONL for MLOps.
    """
    import json
    import time
    import sqlite3
    
    db_path = Path(__file__).parent.parent / "data" / "processed" / "feature_store.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize DB if it doesn't exist
    init_db = not db_path.exists()
    
    max_retries = 5
    retry_delay = 0.2
    for attempt in range(max_retries):
        try:
            with sqlite3.connect(db_path, timeout=5.0) as conn:
                if init_db:
                    cursor = conn.cursor()
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS user_reports (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp INTEGER,
                            exercise TEXT,
                            features_json TEXT,
                            predicted_correct BOOLEAN,
                            actual_correct BOOLEAN,
                            user_feedback TEXT
                        )
                    ''')
                    conn.commit()
                    init_db = False
                    
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO user_reports (timestamp, exercise, features_json, predicted_correct, actual_correct, user_feedback) VALUES (?, ?, ?, ?, ?, ?)",
                    (int(time.time()), body.exercise, json.dumps(body.features), body.predicted_correct, body.actual_correct, body.user_feedback)
                )
                conn.commit()
            break
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            logger.error("Failed to write to feature store: %s", e)
            return {"status": "error"}
        except Exception as e:
            logger.error("Failed to write to feature store: %s", e)
            return {"status": "error"}
        
    logger.info("Data Flywheel report saved to SQLite", exercise=body.exercise, actual_correct=body.actual_correct)
    return {"status": "recorded"}

@app.post("/predict/exercise", response_model=ExercisePrediction)
def predict_exercise_endpoint(request: Request, body: Features13):
    return predict_exercise(body.features)

@app.post("/predict/deadlift", response_model=FormPrediction)
def predict_deadlift_endpoint(request: Request, body: Features13):
    return predict_deadlift(body.features)

@app.post("/predict/benchpress", response_model=FormPrediction)
def predict_benchpress_endpoint(request: Request, body: Features13):
    return predict_benchpress(body.features)

@app.post("/predict/squat", response_model=FormPrediction)
def predict_squat_endpoint(request: Request, body: SquatFeatures):
    return predict_squat(body.model_dump())

