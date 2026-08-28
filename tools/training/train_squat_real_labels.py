"""
Squat Pipeline — Real Human-Labeled Data

Full pipeline:
  1. Extract MediaPipe Pose landmarks from images
  2. Calculate same 20 features as predictor.py (with P0-2 fix: torso_lean = spine_angle)
  3. Train binary model (Good vs Bad) → squat_form.pkl
  4. Train 3-class model (Good vs Bad Back vs Bad Heel) → experimental
  5. Report HONEST metrics on pre-split test set

Dataset: 3 classes (Good, Bad Back, Bad Heel) — HUMAN LABELED images
"""
import cv2
import numpy as np
import pandas as pd
import joblib
import math
import sys
from pathlib import Path
from collections import Counter

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb

try:
    import mediapipe as mp
except ImportError:
    print("[ERROR] mediapipe not installed. Run: pip install mediapipe")
    sys.exit(1)

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ── Paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODELS_DIR = PROJECT_ROOT / "data" / "models"
DATASET_DIR = PROJECT_ROOT / "data" / "raw" / "new_dataset"

# ── Label mapping ──
# Maps folder names to (binary_label, error_code matching SQUAT_ERROR_MAP)
FOLDER_LABEL_MAP = {
    "Good":     (0, 0),   # binary=correct, error=none
    "good":     (0, 0),
    "Bad back": (1, 2),   # binary=incorrect, error=Forward lean
    "Bad Back": (1, 2),
    "bad back": (1, 2),
    "Bad heel": (1, 4),   # binary=incorrect, error=Heels off ground
    "Bad Heel": (1, 4),
    "bad heel": (1, 4),
}

ERROR_MAP = {0: "Good", 2: "Bad Back (Forward Lean)", 4: "Bad Heel (Heels Off)"}


def calculate_angle(a: tuple, b: tuple, c: tuple) -> float:
    """
    3-point angle calculation — identical to client-side calculateAngle().
    a, b, c are (x, y) tuples. Returns angle at vertex b in degrees.
    """
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = abs(math.degrees(radians))
    if angle > 180:
        angle = 360 - angle
    return angle or 0.0


def extract_features_from_image(image_path: Path, pose) -> dict | None:
    """
    Extract 12 base features from a single image using MediaPipe Pose.
    Feature calculation matches client code (camera/page.tsx:310-337).
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if not results.pose_landmarks:
        return None

    lm = results.pose_landmarks.landmark

    # Landmark indices matching client (MediaPipe standard)
    l_shoulder = (lm[11].x, lm[11].y)
    r_shoulder = (lm[12].x, lm[12].y)
    l_hip      = (lm[23].x, lm[23].y)
    r_hip      = (lm[24].x, lm[24].y)
    l_knee     = (lm[25].x, lm[25].y)
    r_knee     = (lm[26].x, lm[26].y)
    l_ankle    = (lm[27].x, lm[27].y)
    r_ankle    = (lm[28].x, lm[28].y)
    l_foot     = (lm[31].x, lm[31].y)
    r_foot     = (lm[32].x, lm[32].y)

    mid_hip = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
    mid_shoulder = ((l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2)
    vertical = (mid_hip[0], mid_hip[1] - 1.0)

    # Calculate angles — same formulas as camera/page.tsx
    spine_angle      = calculate_angle(vertical, mid_hip, mid_shoulder)
    left_knee_angle  = calculate_angle(l_hip, l_knee, l_ankle)
    right_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)
    left_hip_angle   = calculate_angle(l_shoulder, l_hip, l_knee)
    right_hip_angle  = calculate_angle(r_shoulder, r_hip, r_knee)
    left_ankle_angle = calculate_angle(l_knee, l_ankle, l_foot)
    right_ankle_angle = calculate_angle(r_knee, r_ankle, r_foot)

    # Non-angle features — same as client
    left_knee_lateral  = l_knee[0] - l_ankle[0]
    right_knee_lateral = r_ankle[0] - r_knee[0]
    symmetry_score = abs(left_knee_angle - right_knee_angle) + abs(left_hip_angle - right_hip_angle)
    hip_depth = mid_hip[1]

    # P0-2 fix: torso_lean = spine_angle (matches serving)
    torso_lean = spine_angle

    return {
        'left_knee_angle': left_knee_angle,
        'right_knee_angle': right_knee_angle,
        'left_hip_angle': left_hip_angle,
        'right_hip_angle': right_hip_angle,
        'left_ankle_angle': left_ankle_angle,
        'right_ankle_angle': right_ankle_angle,
        'spine_angle': spine_angle,
        'torso_lean': torso_lean,
        'left_knee_lateral': left_knee_lateral,
        'right_knee_lateral': right_knee_lateral,
        'symmetry_score': symmetry_score,
        'hip_depth': hip_depth,
    }


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 8 engineered features — same as predictor.py engineer_squat_features()."""
    df = df.copy()
    df['avg_knee_angle']   = (df['left_knee_angle'] + df['right_knee_angle']) / 2
    df['avg_hip_angle']    = (df['left_hip_angle'] + df['right_hip_angle']) / 2
    df['knee_hip_ratio']   = df['avg_knee_angle'] / (df['avg_hip_angle'] + 1e-8)
    df['knee_depth_ratio'] = df['avg_knee_angle'] / 90.0
    df['ankle_asymmetry']  = (df['left_ankle_angle'] - df['right_ankle_angle']).abs()
    df['hip_asymmetry']    = (df['left_hip_angle'] - df['right_hip_angle']).abs()
    df['total_lateral']    = df['left_knee_lateral'].abs() + df['right_knee_lateral'].abs()
    df['lean_consistency'] = (df['spine_angle'] - df['torso_lean']).abs()
    return df


def get_feature_cols() -> list:
    base = [
        'left_knee_angle', 'right_knee_angle',
        'left_hip_angle', 'right_hip_angle',
        'left_ankle_angle', 'right_ankle_angle',
        'spine_angle', 'torso_lean',
        'left_knee_lateral', 'right_knee_lateral',
        'symmetry_score', 'hip_depth',
    ]
    eng = [
        'avg_knee_angle', 'avg_hip_angle', 'knee_hip_ratio',
        'knee_depth_ratio', 'ankle_asymmetry', 'hip_asymmetry',
        'total_lateral', 'lean_consistency',
    ]
    return base + eng


def extract_dataset(split: str, pose) -> pd.DataFrame:
    """Extract features from all images in a split (train/test)."""
    split_dir = DATASET_DIR / split
    rows = []
    failed = 0
    total = 0

    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        folder_name = class_dir.name
        label_info = FOLDER_LABEL_MAP.get(folder_name)
        if label_info is None:
            print(f"  [WARN] Unknown class folder: {folder_name}")
            continue

        binary_label, error_code = label_info

        images = sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png"))
        print(f"  {folder_name}: {len(images)} images...", end=" ", flush=True)

        class_ok = 0
        for img_path in images:
            total += 1
            feats = extract_features_from_image(img_path, pose)
            if feats is None:
                failed += 1
                continue

            feats['binary_label'] = binary_label
            feats['error_code'] = error_code
            feats['class_name'] = folder_name
            feats['image_file'] = img_path.name
            feats['split'] = split
            rows.append(feats)
            class_ok += 1

        print(f"{class_ok}/{len(images)} OK")

    print(f"  Total: {len(rows)}/{total} extracted ({failed} failed)")
    return pd.DataFrame(rows)


def train_binary_model(X_train, y_train, X_test, y_test):
    """Train binary (Good vs Bad) classifier."""
    print("\n" + "=" * 60)
    print("[TRAIN] Binary — Good vs Bad (REAL human labels)")
    print("=" * 60)

    print(f"  Train: {len(X_train)} ({Counter(y_train)})")
    print(f"  Test:  {len(X_test)} ({Counter(y_test)})")

    # SMOTE if imbalanced
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"  After SMOTE: {len(X_res)} ({Counter(y_res)})")

    m_xgb = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=42, n_jobs=-1)
    m_rf = RandomForestClassifier(
        n_estimators=150, max_depth=15, min_samples_leaf=3,
        class_weight='balanced', random_state=42, n_jobs=-1)

    ensemble = VotingClassifier(
        estimators=[('xgb', m_xgb), ('rf', m_rf)], voting='soft')

    print("  Training VotingClassifier (XGB + RF)...")
    ensemble.fit(X_res, y_res)

    y_pred = ensemble.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n  Accuracy: {acc:.4f}  F1-macro: {f1:.4f}")
    print(f"  Confusion matrix:\n{cm}")
    print(classification_report(y_test, y_pred, target_names=['Good', 'Bad']))

    return ensemble, acc, f1, cm


def train_multiclass_model(X_train, y_train, X_test, y_test):
    """Train 3-class (Good / Bad Back / Bad Heel) classifier with REAL labels."""
    print("\n" + "=" * 60)
    print("[TRAIN] 3-Class — Good / Bad Back / Bad Heel (REAL labels!)")
    print("=" * 60)

    print(f"  Train: {len(X_train)} ({Counter(y_train)})")
    print(f"  Test:  {len(X_test)} ({Counter(y_test)})")

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"  After SMOTE: {len(X_res)} ({Counter(y_res)})")

    m_xgb = xgb.XGBClassifier(
        n_estimators=300, max_depth=8, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='mlogloss', random_state=42, n_jobs=-1)
    m_rf = RandomForestClassifier(
        n_estimators=200, max_depth=20, min_samples_leaf=2,
        class_weight='balanced', random_state=42, n_jobs=-1)

    ensemble = VotingClassifier(
        estimators=[('xgb', m_xgb), ('rf', m_rf)], voting='soft')

    print("  Training VotingClassifier (XGB + RF)...")
    ensemble.fit(X_res, y_res)

    y_pred = ensemble.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)

    class_names = ['Good', 'Bad Back', 'Bad Heel']
    print(f"\n  Accuracy: {acc:.4f}  F1-macro: {f1:.4f}")
    print(f"  Confusion matrix:\n{cm}")
    print(classification_report(y_test, y_pred, target_names=class_names))

    return ensemble, acc, f1, cm


def main():
    print("=" * 60)
    print("[PIPELINE] Squat Model — Real Human-Labeled Images")
    print("=" * 60)

    if not DATASET_DIR.exists():
        print(f"[ERROR] Dataset not found: {DATASET_DIR}")
        return

    # ── Step 1: Extract features ──
    print("\n[STEP 1] Extracting MediaPipe landmarks from images...")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5,
    )

    print("\n--- Train set ---")
    df_train = extract_dataset("train", pose)
    print("\n--- Test set ---")
    df_test = extract_dataset("test", pose)

    pose.close()

    if df_train.empty or df_test.empty:
        print("[ERROR] No features extracted!")
        return

    # ── Step 2: Engineer features ──
    print("\n[STEP 2] Engineering features (20 total)...")
    df_train = add_engineered_features(df_train)
    df_test = add_engineered_features(df_test)

    feat_cols = get_feature_cols()
    print(f"  Features: {feat_cols}")

    # Save extracted features for reproducibility
    csv_path = PROJECT_ROOT / "data" / "processed" / "squat_real_labels.csv"
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    df_all.to_csv(csv_path, index=False)
    print(f"  Saved features CSV: {csv_path.name} ({len(df_all)} rows)")

    X_train = df_train[feat_cols].fillna(0).values
    X_test = df_test[feat_cols].fillna(0).values

    # ── Step 3: Train binary model ──
    y_train_bin = df_train['binary_label'].values
    y_test_bin = df_test['binary_label'].values

    binary_model, bin_acc, bin_f1, bin_cm = train_binary_model(
        X_train, y_train_bin, X_test, y_test_bin)

    # ── Step 4: Train 3-class model ──
    y_train_multi = df_train['error_code'].values
    y_test_multi = df_test['error_code'].values

    multi_model, multi_acc, multi_f1, multi_cm = train_multiclass_model(
        X_train, y_train_multi, X_test, y_test_multi)

    # ── Step 5: Save models ──
    print("\n" + "=" * 60)
    print("[SAVE] Models")
    print("=" * 60)

    # Binary model → squat_form.pkl
    binary_path = MODELS_DIR / "squat_form.pkl"
    backup_path = MODELS_DIR / "squat_form_pre_realdata.pkl"
    if binary_path.exists() and not backup_path.exists():
        import shutil
        shutil.copy2(binary_path, backup_path)
        print(f"  Backed up old model → {backup_path.name}")

    joblib.dump({
        'model': binary_model,
        'feature_cols': feat_cols,
        'model_type': 'binary_real_human_labels',
        'accuracy': bin_acc,
        'f1_macro': bin_f1,
        'confusion_matrix': bin_cm.tolist(),
        'data_source': 'Dataset.zip — human-labeled squat images',
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'classes': {0: 'Good', 1: 'Bad'},
    }, binary_path)
    print(f"  Binary:  {binary_path.name} ({binary_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # 3-class model → squat_form_3class.pkl (experimental — NOT squat_form_detailed)
    multi_path = MODELS_DIR / "squat_form_3class.pkl"
    joblib.dump({
        'model': multi_model,
        'feature_cols': feat_cols,
        'model_type': 'multiclass_3_real_human_labels',
        'accuracy': multi_acc,
        'f1_macro': multi_f1,
        'confusion_matrix': multi_cm.tolist(),
        'data_source': 'Dataset.zip — human-labeled squat images',
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'classes': {0: 'Good', 2: 'Bad Back (Forward Lean)', 4: 'Bad Heel (Heels Off)'},
        'label_map': {0: 'Good', 2: 'Bad Back', 4: 'Bad Heel'},
    }, multi_path)
    print(f"  3-Class: {multi_path.name} ({multi_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"[SUMMARY] Real Human-Labeled Squat Models")
    print(f"{'=' * 60}")
    print(f"  Images processed:  {len(df_all)}")
    print(f"  Train/Test split:  {len(df_train)}/{len(df_test)} (pre-split from dataset)")
    print(f"")
    print(f"  Binary (Good vs Bad):")
    print(f"    Accuracy: {bin_acc:.4f}")
    print(f"    F1-macro: {bin_f1:.4f}")
    print(f"")
    print(f"  3-Class (Good / Bad Back / Bad Heel):")
    print(f"    Accuracy: {multi_acc:.4f}")
    print(f"    F1-macro: {multi_f1:.4f}")
    print(f"")
    print(f"  THESE ARE HONEST NUMBERS — real labels, pre-split test set.")
    print(f"  No data leakage. No pseudo-labels. No circular reasoning.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
