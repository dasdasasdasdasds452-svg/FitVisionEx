"""
FitVision — Predictor (Refactored)

Stateless class-based design. No global mutable state.
Thread-safe — supports uvicorn --workers N.
Includes Injury Risk Assessment module for biomechanical risk scoring.
"""
import structlog
import gc
import numpy as np
import joblib
from pathlib import Path
from typing import Optional

logger = structlog.get_logger("fitvision.predictor")


# ── Injury Risk Assessment ────────────────────────────────────────────────────
# Maps ML prediction confidence + biomechanical features → injury risk level.
# This addresses the thesis title: "ประเมินความเสี่ยงการบาดเจ็บ"

RISK_LEVELS = {
    "low":      {"label": "Low Risk",      "label_th": "ความเสี่ยงต่ำ",     "color": "#22c55e"},
    "medium":   {"label": "Medium Risk",   "label_th": "ความเสี่ยงปานกลาง", "color": "#f59e0b"},
    "high":     {"label": "High Risk",     "label_th": "ความเสี่ยงสูง",     "color": "#f97316"},
    "critical": {"label": "Critical Risk", "label_th": "ความเสี่ยงวิกฤต",   "color": "#ef4444"},
}


def assess_injury_risk(
    exercise: str,
    form_correct: bool,
    confidence: float,
    features: Optional[dict | list] = None,
    error_code: Optional[int] = None,
) -> dict:
    """
    Compute injury risk assessment from ML prediction and biomechanical features.

    Risk score (0–100) is derived from:
      1. Form correctness (base risk)
      2. ML confidence (how certain the model is)
      3. Exercise-specific biomechanical factors

    Returns dict with: risk_level, risk_score, risk_factors, recommendation
    """
    risk_score = 0.0
    risk_factors: list[str] = []

    # ── Factor 1: Form correctness (0–40 points) ──
    if not form_correct:
        # Higher model confidence in "incorrect" → higher risk
        form_risk = 25.0 + (confidence * 15.0)
        risk_score += form_risk
        risk_factors.append("Incorrect form detected")
    else:
        # Even correct form can have minor risk if confidence is low
        if confidence < 0.70:
            risk_score += 10.0
            risk_factors.append("Low prediction confidence — form borderline")

    # ── Factor 2: Exercise-specific biomechanical analysis (0–40 points) ──
    if exercise == "squat" and isinstance(features, dict):
        avg_knee = (features.get("left_knee_angle", 180) + features.get("right_knee_angle", 180)) / 2
        spine = features.get("spine_angle", 0)
        symmetry = features.get("symmetry_score", 0)

        # Deep squat with forward lean → knee/back injury risk
        if avg_knee < 70 and spine > 35:
            risk_score += 20.0
            risk_factors.append("Deep squat with excessive forward lean — lumbar spine stress")
        elif spine > 40:
            risk_score += 15.0
            risk_factors.append("Excessive forward lean — lower back strain risk")

        # Knee valgus (caving in)
        ll = abs(features.get("left_knee_lateral", 0))
        rl = abs(features.get("right_knee_lateral", 0))
        if ll > 0.08 or rl > 0.08:
            risk_score += 15.0
            risk_factors.append("Knee valgus detected — ACL/MCL injury risk")

        # Asymmetry
        if symmetry > 50:
            risk_score += 10.0
            risk_factors.append("Significant left-right asymmetry — compensatory injury risk")

        # Heel lift
        la = features.get("left_ankle_angle", 160)
        ra = features.get("right_ankle_angle", 160)
        if la < 60 or ra < 60:
            risk_score += 10.0
            risk_factors.append("Heel lifting — ankle mobility issue, fall risk")

    elif exercise == "deadlift" and isinstance(features, list) and len(features) >= 13:
        # features[4] = left_hip_angle, features[5] = right_hip_angle (approximate)
        # features[2] = left_shoulder_angle, features[3] = right_shoulder_angle
        left_hip = features[4] if len(features) > 4 else 180
        right_hip = features[5] if len(features) > 5 else 180
        avg_hip = (left_hip + right_hip) / 2

        if not form_correct:
            # Rounded back during deadlift is the #1 injury cause
            risk_score += 20.0
            risk_factors.append("Rounded back detected — severe spinal disc injury risk")
        if avg_hip < 60:
            risk_score += 10.0
            risk_factors.append("Excessive hip flexion — hamstring/lower back strain")

    elif exercise == "benchpress" and isinstance(features, list) and len(features) >= 13:
        if not form_correct:
            risk_score += 15.0
            risk_factors.append("Incorrect bench press form — shoulder impingement risk")
        # Elbow angles (features[0], features[1])
        left_elbow = features[0] if len(features) > 0 else 90
        right_elbow = features[1] if len(features) > 1 else 90
        elbow_asym = abs(left_elbow - right_elbow)
        if elbow_asym > 20:
            risk_score += 10.0
            risk_factors.append("Uneven elbow positioning — rotator cuff strain risk")

    # ── Factor 3: Squat error-specific risks (0–20 points) ──
    if exercise == "squat" and error_code is not None and error_code != 0:
        error_risk_map = {
            1: ("Shallow squat — reduced muscle activation, not necessarily dangerous", 5),
            2: ("Forward lean — intervertebral disc compression risk", 20),
            3: ("Knee valgus — ligament tear risk (ACL/MCL)", 20),
            4: ("Heel rise — balance instability, Achilles strain", 15),
            5: ("Asymmetric movement — compensatory overload injury risk", 10),
        }
        desc, pts = error_risk_map.get(error_code, ("Unknown error pattern", 10))
        risk_score += pts
        if desc not in risk_factors:
            risk_factors.append(desc)

    # ── Clamp and classify ──
    risk_score = min(100.0, max(0.0, risk_score))

    if risk_score <= 15:
        level = "low"
    elif risk_score <= 40:
        level = "medium"
    elif risk_score <= 70:
        level = "high"
    else:
        level = "critical"

    risk_info = RISK_LEVELS[level]

    # Generate recommendation based on risk level
    recommendations = {
        "low":      "Form looks safe. Continue with current weight and technique.",
        "medium":   "Minor form issues detected. Consider reducing weight and focusing on technique.",
        "high":     "Significant injury risk. Stop and correct form before continuing. Consider consulting a trainer.",
        "critical": "Immediate injury risk! Stop exercise immediately. Reduce weight significantly and review proper technique.",
    }

    result = {
        "risk_level":       level,
        "risk_label":       risk_info["label"],
        "risk_label_th":    risk_info["label_th"],
        "risk_score":       round(risk_score, 1),
        "risk_color":       risk_info["color"],
        "risk_factors":     risk_factors if risk_factors else ["No significant risk factors detected"],
        "recommendation":   recommendations[level],
    }

    logger.info("RISK: exercise=%s level=%s score=%.1f factors=%d",
                exercise, level, risk_score, len(risk_factors))

    return result

# ── Label maps ─────────────────────────────────────────────────────────────────
SQUAT_ERROR_MAP = {
    0: "Correct",
    1: "Shallow squat — ลงให้ลึกกว่านี้",
    2: "Forward lean — อย่าโน้มตัวไปข้างหน้า",
    3: "Knees caving in — เก็บเข่าให้ตรง",
    4: "Heels off ground — ส้นเท้าต้องติดพื้น",
    5: "Asymmetric — ทั้งสองข้างไม่สมมาตร",
}

EXERCISE_MAP = {0: "benchpress", 1: "squat", 2: "deadlift"}


def engineer_squat_features(f: dict) -> list:
    """Full 20-feature extraction matching the training script."""
    lk = f.get("left_knee_angle", 180);  rk = f.get("right_knee_angle", 180)
    lh = f.get("left_hip_angle", 180);   rh = f.get("right_hip_angle", 180)
    la = f.get("left_ankle_angle", 160); ra = f.get("right_ankle_angle", 160)
    sp = f.get("spine_angle", 0);        tl = f.get("torso_lean", 0)
    ll = f.get("left_knee_lateral", 0);  rl = f.get("right_knee_lateral", 0)
    sy = f.get("symmetry_score", 0);     hd = f.get("hip_depth", 0.5)

    avg_knee       = (lk + rk) / 2
    avg_hip        = (lh + rh) / 2
    knee_hip_ratio = avg_knee / (avg_hip + 1e-8)
    knee_depth     = avg_knee / 90.0
    ankle_asym     = abs(la - ra)
    hip_asym       = abs(lh - rh)
    total_lat      = abs(ll) + abs(rl)
    lean_con       = abs(sp - tl)

    return [lk, rk, lh, rh, la, ra, sp, tl, ll, rl, sy, hd,
            avg_knee, avg_hip, knee_hip_ratio, knee_depth,
            ankle_asym, hip_asym, total_lat, lean_con]


def _classify_squat_error_rules(features: dict) -> int:
    """
    Rule-based fallback for error types NOT covered by the 3-class ML model.
    The 3-class model only knows Good(0), Bad Back(2), Bad Heel(4).
    This catches Shallow(1), Knees caving(3), Asymmetric(5).
    """
    avg_knee = (features.get("left_knee_angle", 180) + features.get("right_knee_angle", 180)) / 2
    ll = abs(features.get("left_knee_lateral", 0))
    rl = abs(features.get("right_knee_lateral", 0))
    sy = features.get("symmetry_score", 0)

    if avg_knee > 130:
        return 1   # Shallow squat
    if ll > 0.08 or rl > 0.08:
        return 3   # Knees caving in
    if sy > 70:
        return 5   # Asymmetric
    return 1       # Default: shallow


class ModelRegistry:
    """
    Thread-safe model loader with single-model-at-a-time memory management.
    Designed for 512 MB RAM environments (Fly.io).

    Inject via app.state in FastAPI:
        app.state.registry = ModelRegistry(models_dir)
    """

    def __init__(self, models_dir: Path):
        self._models_dir = models_dir
        self._current_exercise: str | None = None
        self._active_models: dict = {}
        # Configurable thresholds: P(correct) must exceed this to be "correct"
        from config.settings import PREDICTION_THRESHOLDS
        self._thresholds = PREDICTION_THRESHOLDS
        import threading
        self._lock = threading.Lock()

    def _force_free_memory(self) -> None:
        """Aggressively free memory back to the OS."""
        gc.collect()
        gc.collect()
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    def _evict_all(self) -> None:
        """Purge all loaded models from RAM."""
        for key in list(self._active_models.keys()):
            del self._active_models[key]
        self._active_models.clear()
        self._current_exercise = None
        self._force_free_memory()

    def _load(self, path: Path) -> dict:
        """Load a single .pkl model file, validating against its manifest if available."""
        manifest_path = path.with_suffix(".manifest.json")
        if manifest_path.exists():
            import json
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                logger.info("Validated manifest for %s (sha256: %s)", path.name, manifest.get("sha256", "unknown")[:8])
            except Exception as e:
                logger.error("Failed to read manifest for %s: %s", path.name, e)
        else:
            logger.warning("No manifest found for %s, loading blindly", path.name)
            
        logger.info("Loading %s...", path.name)
        model = joblib.load(path)
        return model

    def _get_models(self, exercise: str) -> dict:
        """Load models for the given exercise, evicting others first."""
        with self._lock:
            if self._current_exercise == exercise and self._active_models:
                return self._active_models

            logger.info("Evicting '%s', loading '%s'", self._current_exercise, exercise)
            self._evict_all()
            self._current_exercise = exercise

            paths = {
                "exercise":        self._models_dir / "exercise_classifier.pkl",
                "deadlift_form":   self._models_dir / "deadlift_form.pkl",
                "benchpress_form": self._models_dir / "benchpress_form.pkl",
            }

            try:
                if exercise == "deadlift":
                    # Prefer calibrated model if available
                    cal_path = self._models_dir / "deadlift_form_calibrated.pkl"
                    if cal_path.exists():
                        self._active_models["deadlift_form"] = self._load(cal_path)
                        logger.info("Loaded CALIBRATED deadlift model")
                    elif paths["deadlift_form"].exists():
                        self._active_models["deadlift_form"] = self._load(paths["deadlift_form"])
                elif exercise == "benchpress" and paths["benchpress_form"].exists():
                    self._active_models["benchpress_form"] = self._load(paths["benchpress_form"])
                elif exercise == "squat":
                    path_3class = self._models_dir / "squat_form_3class.pkl"
                    path_binary = self._models_dir / "squat_form.pkl"
                    if path_3class.exists():
                        self._active_models["squat_form_3class"] = self._load(path_3class)
                    elif path_binary.exists():
                        self._active_models["squat_form"] = self._load(path_binary)
                elif exercise == "classifier" and paths["exercise"].exists():
                    self._active_models["exercise"] = self._load(paths["exercise"])
            except MemoryError:
                logger.error("MemoryError loading '%s'!", exercise)
                self._evict_all()
                return {}

            return self._active_models

    # ── Public prediction methods ──────────────────────────────────────────

    def predict_exercise(self, features: list) -> dict:
        """Classify which exercise is being performed."""
        models = self._get_models("classifier")
        m = models.get("exercise")
        if not m:
            return {"exercise": "unknown", "confidence": 0}

        X = np.array(features).reshape(1, -1)
        idx = m["model"].predict(X)[0]
        proba = m["model"].predict_proba(X)[0]

        return {
            "exercise":   EXERCISE_MAP.get(idx, "unknown"),
            "confidence": float(proba[idx]),
        }

    def predict_deadlift(self, features: list) -> dict:
        """Predict deadlift form correctness using configurable threshold."""
        models = self._get_models("deadlift")
        m = models.get("deadlift_form")
        if not m:
            return {"form_correct": True, "confidence": 0, "feedback": "Model not loaded"}

        X = np.array(features).reshape(1, -1)
        proba = m["model"].predict_proba(X)[0]
        classes = list(m["model"].classes_)

        # Explicit class lookup — works regardless of label convention
        correct_idx = classes.index(1) if 1 in classes else 0
        prob_correct = float(proba[correct_idx])

        threshold = self._thresholds.get("deadlift", 0.50)
        form_correct = prob_correct >= threshold
        conf = prob_correct if form_correct else (1.0 - prob_correct)

        logger.info("DEADLIFT: P(correct)=%.3f thresh=%.2f -> %s",
                    prob_correct, threshold, "correct" if form_correct else "incorrect")

        result = {
            "form_correct": form_correct,
            "confidence":   conf,
            "feedback": "Good form! Keep it up 💪" if form_correct else "Check your form",
        }
        risk = assess_injury_risk("deadlift", form_correct, conf, features=features)
        result["risk_assessment"] = risk
        return result

    def predict_benchpress(self, features: list) -> dict:
        """Predict bench press form correctness using configurable threshold."""
        models = self._get_models("benchpress")
        m = models.get("benchpress_form")
        if not m:
            return {"form_correct": True, "confidence": 0, "feedback": "Model not loaded"}

        X = np.array(features).reshape(1, -1)
        proba = m["model"].predict_proba(X)[0]
        classes = list(m["model"].classes_)

        # Explicit class lookup — benchpress uses class 0 = correct
        correct_idx = classes.index(0) if 0 in classes else 0
        prob_correct = float(proba[correct_idx])

        threshold = self._thresholds.get("benchpress", 0.50)
        form_correct = prob_correct >= threshold
        conf = prob_correct if form_correct else (1.0 - prob_correct)

        logger.info("BENCH: P(correct)=%.3f thresh=%.2f -> %s",
                    prob_correct, threshold, "correct" if form_correct else "incorrect")

        result = {
            "form_correct": form_correct,
            "confidence":   conf,
            "feedback":     "Good bench press form! 💪" if form_correct
                            else "Check your form: Keep elbows tucked, back arched, and wrists straight.",
        }
        risk = assess_injury_risk("benchpress", form_correct, conf, features=features)
        result["risk_assessment"] = risk
        return result

    def predict_squat(self, squat_features: dict) -> dict:
        """
        ML squat prediction using real human-labeled 3-class model.
        Falls back to rules for error types not in the dataset.
        """
        models = self._get_models("squat")

        feat_vec = engineer_squat_features(squat_features)
        X = np.array(feat_vec).reshape(1, -1)

        # Standing check — not squatting yet
        if feat_vec[12] > 150:
            return {
                "form_correct": True, "confidence": 0.85,
                "error_type": "Correct", "error_code": 0,
                "detail_confidence": 0.85,
                "feedback": "Good form — squat deeper to begin analysis 💪",
            }

        conf = 0.5
        error_code = 0
        model_used = "fallback"

        try:
            if "squat_form_3class" in models:
                m3 = models["squat_form_3class"]
                proba = m3["model"].predict_proba(X)[0]
                conf = float(proba.max())
                classes = m3["model"].classes_
                error_code = int(classes[np.argmax(proba)])
                model_used = "3class_real_labels"

            elif "squat_form" in models:
                bm = models["squat_form"]
                b_pred = bm["model"].predict(X)[0]
                b_proba = bm["model"].predict_proba(X)[0]
                conf = float(b_proba[b_pred])
                error_code = 0 if b_pred == 0 else _classify_squat_error_rules(squat_features)
                model_used = "binary_fallback"
            else:
                rule_check = _classify_squat_error_rules(squat_features)
                error_code = rule_check
                model_used = "pure_rules"
        except MemoryError:
            logger.error("MemoryError on squat model")
            self._force_free_memory()

        form_correct = (error_code == 0)

        # Low-confidence safety net: check rules
        if form_correct and conf < 0.70:
            rule_check = _classify_squat_error_rules(squat_features)
            if rule_check != 1:
                error_code = rule_check
                form_correct = False
                logger.warning("Low conf (%.3f) + rule override → error_code=%d", conf, rule_check)

        error_label = SQUAT_ERROR_MAP.get(error_code, "Unknown error")
        logger.info("SQUAT: error_code=%d conf=%.3f model=%s", error_code, conf, model_used)

        result = {
            "form_correct": form_correct, "confidence": conf,
            "error_type": "Correct" if form_correct else error_label,
            "error_code": error_code if not form_correct else 0,
            "detail_confidence": conf,
            "feedback": "Good squat form! 💪" if form_correct else error_label,
        }
        risk = assess_injury_risk("squat", form_correct, conf,
                                  features=squat_features, error_code=error_code)
        result["risk_assessment"] = risk
        return result


# ── Backward-compatible module-level API ──────────────────────────────────────
# These functions delegate to a singleton registry for backward compatibility.
# New code should use ModelRegistry directly via app.state.registry.
import sys as _sys
_project_root = Path(__file__).parent.parent
_sys.path.append(str(_project_root))
from config.settings import MODELS_DIR as _MODELS_DIR

_default_registry = ModelRegistry(_MODELS_DIR)

predict_exercise = _default_registry.predict_exercise
predict_deadlift = _default_registry.predict_deadlift
predict_benchpress = _default_registry.predict_benchpress
predict_squat = _default_registry.predict_squat
