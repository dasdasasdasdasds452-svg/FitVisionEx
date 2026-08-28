commit 044bdedda7ee51c968ca942c2534f1e0d8ee48ab
Author: Aussadawut Runekew <aussadawut.r@kkumail.com>
Date:   Sun Mar 8 04:52:18 2026 +0700

    feat: replace rules with custom machine learning models

diff --git a/app/predictor.py b/app/predictor.py
index 581ead4..3e8f662 100644
--- a/app/predictor.py
+++ b/app/predictor.py
@@ -23,41 +23,45 @@ SQUAT_ERROR_MAP = {
 
 EXERCISE_MAP = {0: "benchpress", 1: "squat", 2: "deadlift"}
 
-# ── Squat Rule-Based Prediction ────────────────────────────────────────────────
-# The Kaggle training dataset computes angles differently from our MediaPipe
-# frontend (e.g., spine_angle: Kaggle standing=149°, frontend standing=7°).
-# ML models trained on Kaggle data cannot work with frontend features.
-# Solution: Use rule-based analysis on raw frontend values directly.
+def engineer_squat_features(f: dict) -> list:
+    """Full 20-feature extraction matching the v7 powerlifting training script."""
+    lk = f.get("left_knee_angle", 180);  rk = f.get("right_knee_angle", 180)
+    lh = f.get("left_hip_angle", 180);   rh = f.get("right_hip_angle", 180)
+    la = f.get("left_ankle_angle", 160); ra = f.get("right_ankle_angle", 160)
+    sp = f.get("spine_angle", 0);        tl = f.get("torso_lean", 0)
+    ll = f.get("left_knee_lateral", 0);  rl = f.get("right_knee_lateral", 0)
+    sy = f.get("symmetry_score", 0);     hd = f.get("hip_depth", 0.5)
+
+    avg_knee       = (lk + rk) / 2
+    avg_hip        = (lh + rh) / 2
+    knee_hip_ratio = avg_knee / (avg_hip + 1e-8)
+    knee_depth     = avg_knee / 90.0
+    ankle_asym     = abs(la - ra)
+    hip_asym       = abs(lh - rh)
+    total_lat      = abs(ll) + abs(rl)
+    lean_con       = abs(sp - tl)
+
+    # MUST match get_feature_cols() in train_powerlifting_ml.py exact order
+    return [lk, rk, lh, rh, la, ra, sp, tl, ll, rl, sy, hd,
+            avg_knee, avg_hip, knee_hip_ratio, knee_depth,
+            ankle_asym, hip_asym, total_lat, lean_con]
 
-def predict_squat_rules(f: dict) -> dict:
+def predict_squat(squat_features: dict) -> dict:
     """
-    Rule-based squat form analysis using raw MediaPipe angles from frontend.
+    ⚡ MEMORY-OPTIMIZED ML SQUAT PREDICTION ⚡
     
-    Thresholds derived from exercise science + calibrated with debug output:
-    - Frontend spine_angle: 0° = perfectly upright, higher = more lean
-    - Frontend knee_angle: 180° = straight legs, lower = deeper squat
-    - Frontend hip_depth: 0.5 = standing, 0.8+ = deep squat (normalized y)
+    Loads each squat model ONE AT A TIME to avoid OOM on 512MB instances.
+    Uses the new perfectly-matched v7 models trained on Powerlifting raw landmarks.
     """
-    lk = f.get("left_knee_angle", 180)
-    rk = f.get("right_knee_angle", 180)
-    sp = f.get("spine_angle", 0)
-    ll = f.get("left_knee_lateral", 0)
-    rl = f.get("right_knee_lateral", 0)
-    sy = f.get("symmetry_score", 0)
-    hd = f.get("hip_depth", 0.5)
-    la = f.get("left_ankle_angle", 160)
-    ra = f.get("right_ankle_angle", 160)
-    lh = f.get("left_hip_angle", 180)
-    rh = f.get("right_hip_angle", 180)
-    
-    avg_knee = (lk + rk) / 2
+    # First, evict any other models from RAM
+    _evict_all_models()
     
-    errors = []
-    confidence_penalties = []
+    feat_vec = engineer_squat_features(squat_features)
+    X = np.array(feat_vec).reshape(1, -1)
     
-    # === Check if person is actually squatting (not just standing) ===
-    if avg_knee > 150:
-        # Standing or barely bent — not in squat position yet
+    # Check if they are just standing (avg_knee > 150)
+    # If so, return correct immediately without loading models to save time
+    if feat_vec[12] > 150:
         return {
             "form_correct": True,
             "confidence": 0.85,
@@ -67,219 +71,63 @@ def predict_squat_rules(f: dict) -> dict:
             "feedback": "Good form — squat deeper to begin analysis 💪",
         }
     
-    # === 1. Shallow Squat Check ===
-    # Good squat: knee angle < 100° (parallel or below)
-    # Acceptable: 100-120°
-    # Shallow: > 120°
-    if avg_knee > 130:
-        errors.append((1, "Shallow squat — ลงให้ลึกกว่านี้", 0.9))
-    elif avg_knee > 115:
-        errors.append((1, "Shallow squat — ลงให้ลึกกว่านี้", 0.7))
-    
-    # === 2. Forward Lean Check ===
-    # Frontend spine_angle: 0° = upright, increases with lean
-    # Good: < 25°, Warning: 25-35°, Bad: > 35°
-    if sp > 40:
-        errors.append((2, "Forward lean — อย่าโน้มตัวไปข้างหน้า", 0.9))
-    elif sp > 30:
-        errors.append((2, "Forward lean — อย่าโน้มตัวไปข้างหน้า", 0.7))
-    
-    # === 3. Knees Caving In Check ===
-    # left_knee_lateral < 0 or right_knee_lateral < 0 means caving
-    # abs > 0.08 is significant
-    total_lateral = abs(ll) + abs(rl)
-    if ll < -0.06 or rl < -0.06:
-        errors.append((3, "Knees caving in — เก็บเข่าให้ตรง", 0.85))
-    elif total_lateral > 0.12:
-        errors.append((3, "Knees caving in — เก็บเข่าให้ตรง", 0.65))
-    
-    # === 4. Heels Off Ground Check ===  
-    # Ankle angle changes when heels lift
-    # Normal ankle angle during squat: 70-120° range
-    # Very high ankle angle (>140°) while squatting suggests heels off
-    avg_ankle = (la + ra) / 2
-    if avg_knee < 120 and avg_ankle > 155:
-        errors.append((4, "Heels off ground — ส้นเท้าต้องติดพื้น", 0.8))
-    
-    # === 5. Asymmetry Check ===
-    # symmetry_score = |left_knee - right_knee| + |left_hip - right_hip|
-    # Note: hip angles vary a lot due to camera angle, so threshold is high
-    if sy > 70:
-        errors.append((5, "Asymmetric — ทั้งสองข้างไม่สมมาตร", 0.85))
-    elif sy > 50:
-        errors.append((5, "Asymmetric — ทั้งสองข้างไม่สมมาตร", 0.65))
-    
-    # === Determine result ===
-    if not errors:
-        # All checks passed → Correct form!
-        base_conf = 0.85
-        # Higher confidence for deeper, more stable squats
-        if avg_knee < 100 and sp < 20:
-            base_conf = 0.92
-        
-        return {
-            "form_correct": True,
-            "confidence": base_conf,
-            "error_type": "Correct",
-            "error_code": 0,
-            "detail_confidence": base_conf,
-            "feedback": "Good squat form! 💪",
-        }
-    else:
-        # Pick the error with highest confidence
-        errors.sort(key=lambda x: x[2], reverse=True)
-        top_error = errors[0]
-        error_code, error_label, error_conf = top_error
-        
-        print(f"[FitVision] 🔍 RULES: avg_knee={avg_knee:.1f}° spine={sp:.1f}° "
-              f"lat=({ll:.3f},{rl:.3f}) sym={sy:.1f} ankle={avg_ankle:.1f}")
-        print(f"[FitVision] 🔍 ERRORS: {[(e[0], e[2]) for e in errors]}")
-        
-        return {
-            "form_correct": False,
-            "confidence": error_conf,
-            "error_type": error_label,
-            "error_code": error_code,
-            "detail_confidence": error_conf,
-            "feedback": error_label,
-        }
-
-
-# ── Model loader ───────────────────────────────────────────────────────────────
-_current_exercise = None
-_active_models = {}
-
-def _force_free_memory():
-    """Aggressively free memory back to the OS."""
-    gc.collect()
-    gc.collect()
-    try:
-        import ctypes
-        ctypes.CDLL("libc.so.6").malloc_trim(0)
-    except Exception:
-        pass
-
-def _log_memory():
-    """Log current memory usage for debugging OOM."""
-    try:
-        import os
-        rss = os.popen('cat /proc/self/status | grep VmRSS').read().strip()
-        if rss:
-            print(f"[FitVision] 📊 {rss}")
-    except Exception:
-        pass
-
-def _evict_all_models():
-    """Purge all loaded models from RAM."""
-    global _current_exercise, _active_models
-    for key in list(_active_models.keys()):
-        obj = _active_models.pop(key)
-        del obj
-    _active_models.clear()
-    _current_exercise = None
-    _force_free_memory()
-
-def _load_single_model(path):
-    """Load a single model file, with memory logging."""
-    _log_memory()
-    print(f"[FitVision] 📦 Loading {path.name}...")
-    model = joblib.load(path)
-    _log_memory()
-    return model
-
-def get_exercise_models(exercise: str):
-    """
-    Strict Memory Management for 512MB RAM Limit.
-    Squat is now rule-based — no model loading needed!
-    """
-    global _current_exercise, _active_models
-    
-    if _current_exercise == exercise and _active_models:
-        return _active_models
-    
-    print(f"\n[FitVision] 🧹 RAM Cleanup: Evicting models for '{_current_exercise}'...")
-    _evict_all_models()
-    
-    _current_exercise = exercise
-    
-    paths = {
-        "exercise":       MODELS_DIR / "exercise_classifier.pkl",
-        "deadlift_form":  MODELS_DIR / "deadlift_form.pkl",
-        "benchpress_form":MODELS_DIR / "benchpress_form.pkl",
-    }
+    squat_binary_path   = MODELS_DIR / "squat_form.pkl"
+    squat_detailed_path = MODELS_DIR / "squat_form_detailed.pkl"
     
+    # ── Stage 1: Binary classification (Correct / Incorrect) ──
+    b_pred = 0
+    b_proba = [1.0, 0.0]
     try:
-        if exercise == "deadlift":
-            if paths["deadlift_form"].exists():
-                _active_models["deadlift_form"] = _load_single_model(paths["deadlift_form"])
-        elif exercise == "benchpress":
-            if paths["benchpress_form"].exists():
-                _active_models["benchpress_form"] = _load_single_model(paths["benchpress_form"])
-        elif exercise == "classifier":
-            if paths["exercise"].exists():
-                _active_models["exercise"] = _load_single_model(paths["exercise"])
+        if squat_binary_path.exists():
+            bm = _load_single_model(squat_binary_path)
+            b_pred  = bm["model"].predict(X)[0]
+            b_proba = bm["model"].predict_proba(X)[0]
+            # FREE immediately
+            del bm
+            _force_free_memory()
+            print("[FitVision] ✅ Binary model done")
     except MemoryError:
-        print(f"[FitVision] ❌ MemoryError loading '{exercise}'!")
-        _evict_all_models()
-        return {}
-        
-    print(f"[FitVision] ✅ Model swap complete.")
-    return _active_models
-
-
-# ── Predict functions ──────────────────────────────────────────────────────────
-def predict_exercise(features: list) -> dict:
-    models = get_exercise_models("classifier")
-    m = models.get("exercise")
-    if not m:
-        return {"exercise": "unknown", "confidence": 0}
-
-    X   = np.array(features).reshape(1, -1)
-    idx = m["model"].predict(X)[0]
-    proba = m["model"].predict_proba(X)[0]
-
-    return {
-        "exercise":   EXERCISE_MAP.get(idx, "unknown"),
-        "confidence": float(proba[idx]),
-    }
-
-
-def predict_deadlift(features: list) -> dict:
-    models = get_exercise_models("deadlift")
-    m = models.get("deadlift_form")
-    if not m:
-        return {"form_correct": True, "confidence": 0, "feedback": "Model not loaded"}
-
-    X     = np.array(features).reshape(1, -1)
-    pred  = m["model"].predict(X)[0]
-    proba = m["model"].predict_proba(X)[0]
-    conf  = float(proba[pred])
-
+        print("[FitVision] ❌ MemoryError on squat binary model")
+        _force_free_memory()
+    
+    form_correct = (b_pred == 0)
+    
+    # ── Stage 2: Multi-class error diagnosis ──
+    # Only load the multiclass model if the form was incorrect
+    d_pred = 0
+    d_proba = [1.0] + [0.0] * 5
+    
+    if not form_correct:
+        try:
+            if squat_detailed_path.exists():
+                dm = _load_single_model(squat_detailed_path)
+                d_pred  = dm["model"].predict(X)[0]
+                d_proba = dm["model"].predict_proba(X)[0]
+                # FREE immediately
+                del dm
+                _force_free_memory()
+                print("[FitVision] ✅ Detailed model done")
+            else:
+                d_pred = 1 # Fallback to shallow
+        except MemoryError:
+            print("[FitVision] ❌ MemoryError on squat detailed model")
+            _force_free_memory()
+            d_pred = 1
+    
+    error_label  = SQUAT_ERROR_MAP.get(d_pred, "Unknown error")
+    
+    print(f"[FitVision] 🔍 ML SQUAT: b_pred={b_pred} d_pred={d_pred} error={error_label}")
+    
     return {
-        "form_correct": bool(pred),
-        "confidence":   conf,
-        "feedback": "Good form! Keep it up" if pred else "Check your form",
+        "form_correct":      form_correct,
+        "confidence":        float(b_proba[b_pred]),
+        "error_type":        "Correct" if form_correct else error_label,
+        "error_code":        int(d_pred) if not form_correct else 0,
+        "detail_confidence": float(d_proba[d_pred]),
+        "feedback":          "Good squat form! 💪" if form_correct else error_label,
     }
 
 
-def predict_squat(squat_features: dict) -> dict:
-    """
-    ⚡ RULE-BASED SQUAT PREDICTION ⚡
-    
-    Uses direct MediaPipe angle values with exercise science thresholds.
-    No model loading = no OOM risk, instant response!
-    """
-    # Evict other models to free RAM when switching to squat
-    _evict_all_models()
-    
-    result = predict_squat_rules(squat_features)
-    
-    print(f"[FitVision] 🔍 SQUAT: form_correct={result['form_correct']} "
-          f"conf={result['confidence']:.2f} error={result['error_type']}")
-    
-    return result
-
-
 def predict_benchpress(features: list) -> dict:
     models = get_exercise_models("benchpress")
     m = models.get("benchpress_form")
