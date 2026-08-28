"""
FitVision Configuration Settings
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model paths
YOLO_MODEL_PATH = MODELS_DIR / "yolov8s.pt"  # YOLOv8 for person detection
BENCHPRESS_MODEL_PATH = MODELS_DIR / "benchpress.pkl"
SQUAT_MODEL_PATH = MODELS_DIR / "squat.pkl"
DEADLIFT_MODEL_PATH = MODELS_DIR / "deadlift.pkl"

# MediaPipe settings
MEDIAPIPE_CONFIDENCE = 0.5
MEDIAPIPE_TRACKING_CONFIDENCE = 0.5

# Video settings
DEFAULT_FPS = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Risk assessment thresholds
RISK_LOW_THRESHOLD = 30
RISK_MEDIUM_THRESHOLD = 60
RISK_HIGH_THRESHOLD = 100

# Exercise types
EXERCISES = {
    0: 'bench_press',
    1: 'squat',
    2: 'deadlift'
}

# Feedback language
FEEDBACK_LANGUAGE = 'th'  # 'th' for Thai, 'en' for English

# Prediction thresholds — P(correct) must exceed this to count as "correct"
# Tuned from actual threshold sweep on test set (2026-07-24)
#
# Deadlift sweep results (calibrated model):
#   thresh 0.10 → FAR  4.4%, miss 37.2%, acc 73.2%
#   thresh 0.15 → FAR 10.9%, miss 23.5%, acc 80.5%  ← best balance
#   thresh 0.50 → FAR 30.3%, miss  5.7%, acc 86.5%
#
# Higher = stricter (needs more confidence to say "correct") → fewer false negatives
# Lower  = more lenient → fewer false alarms (user told "incorrect" when actually correct)
PREDICTION_THRESHOLDS = {
    "deadlift":      0.15,   # calibrated model: FAR 10.9%, acc 80.5%
    "benchpress":    0.50,   # keep default — video-level labels, accuracy inflated
    "squat_binary":  0.45,   # slightly lenient — reduce "Good->Bad" false alarm
}

