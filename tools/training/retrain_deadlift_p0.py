"""
Retrain Deadlift Model — P0-4 Fix

Fixes:
  1. Class imbalance 1:2.78 → class_weight='balanced' + SMOTE
  2. elbow_symmetry/knee_symmetry bug → recompute from angle columns
  3. Data leakage → GroupShuffleSplit by video_name
  4. Per-class metrics (precision/recall/F1)

Produces: deadlift_form.pkl
"""
import pandas as pd
import numpy as np
import joblib
import sys
from pathlib import Path
from collections import Counter

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import xgboost as xgb

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# ── Paths (relative) ──
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODELS_DIR = PROJECT_ROOT / "data" / "models"
DATA_CSV = PROJECT_ROOT / "data" / "processed" / "training_dataset.csv"

# Feature columns in the order the client sends them
FEATURE_COLS = [
    'left_elbow_angle', 'right_elbow_angle',
    'left_shoulder_angle', 'right_shoulder_angle',
    'left_hip_angle', 'right_hip_angle',
    'left_knee_angle', 'right_knee_angle',
    'shoulder_width', 'hip_width', 'torso_length',
    'elbow_symmetry', 'knee_symmetry',
]


def main():
    print("=" * 60)
    print("[P0-4] Retrain Deadlift — Fix imbalance + symmetry bug + leakage")
    print("=" * 60)

    if not DATA_CSV.exists():
        print(f"[ERROR] CSV not found: {DATA_CSV}")
        return

    df = pd.read_csv(DATA_CSV, low_memory=False)
    df_dl = df[df['exercise'] == 'deadlift'].copy()

    print(f"[OK] Loaded {len(df_dl)} deadlift rows")
    print(f"     form_correct: True={int((df_dl['form_correct'] == True).sum())}, "
          f"False={int((df_dl['form_correct'] == False).sum())}")
    print(f"     video_name unique: {df_dl['video_name'].nunique()}")
    print(f"     deadlift_type: {df_dl['deadlift_type'].value_counts().to_dict()}")

    # ── Fix #1: Recompute elbow_symmetry and knee_symmetry ──
    print(f"\n[FIX] elbow_symmetry before: mean={df_dl['elbow_symmetry'].mean():.4f} "
          f"(all zeros = {(df_dl['elbow_symmetry'] == 0).all()})")
    print(f"[FIX] knee_symmetry before:  mean={df_dl['knee_symmetry'].mean():.4f} "
          f"(all zeros = {(df_dl['knee_symmetry'] == 0).all()})")

    # Recompute from the actual angle columns
    df_dl['elbow_symmetry'] = (df_dl['left_elbow_angle'] - df_dl['right_elbow_angle']).abs()
    df_dl['knee_symmetry'] = (df_dl['left_knee_angle'] - df_dl['right_knee_angle']).abs()

    print(f"[FIX] elbow_symmetry after:  mean={df_dl['elbow_symmetry'].mean():.2f}, "
          f"std={df_dl['elbow_symmetry'].std():.2f}")
    print(f"[FIX] knee_symmetry after:   mean={df_dl['knee_symmetry'].mean():.2f}, "
          f"std={df_dl['knee_symmetry'].std():.2f}")

    # ── BUT: Client sends binary 0/1 for symmetry, not continuous ──
    # Client: Math.abs(lm[13].y - lm[14].y) < 0.05 ? 1 : 0
    # This is a DIFFERENT computation. To avoid skew, we have two options:
    #   A) Train on continuous (correct) and fix client to send continuous
    #   B) Train on binary 0/1 (matches current client)
    #
    # We go with A (train on continuous) because:
    # - Continuous values carry more information
    # - We'll fix the client afterwards
    # - The model can learn meaningful thresholds itself
    print("\n[NOTE] Training with continuous symmetry values.")
    print("       Client will need matching fix (see camera/page.tsx:262-263)")

    # Prepare features
    X = df_dl[FEATURE_COLS].apply(pd.to_numeric, errors='coerce').fillna(0).values
    y = df_dl['form_correct'].astype(int).values  # 1=correct, 0=incorrect
    groups = df_dl['video_name'].values

    print(f"\n[DATA] {len(X)} samples, {len(FEATURE_COLS)} features")
    print(f"       Label: Correct(1)={int((y == 1).sum())}, Incorrect(0)={int((y == 0).sum())}")
    print(f"       Ratio: 1:{(y == 0).sum() / max((y == 1).sum(), 1):.2f}")

    # ── Fix #2: GroupShuffleSplit ──
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_videos = set(groups[train_idx])
    test_videos = set(groups[test_idx])
    print(f"\n[SPLIT] GroupShuffleSplit by video_name:")
    print(f"  Train: {len(X_train)} samples, {len(train_videos)} videos")
    print(f"  Test:  {len(X_test)} samples, {len(test_videos)} videos")
    print(f"  Overlap: {train_videos & test_videos}  (should be empty)")
    print(f"  Train labels: {Counter(y_train)}")
    print(f"  Test labels:  {Counter(y_test)}")

    # ── Fix #3: SMOTE + class_weight='balanced' ──
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"\n[SMOTE] After: {len(X_res)} samples ({Counter(y_res)})")

    # ── Train ensemble ──
    from mlflow_tracker import MLflowTracker
    print("\n[TRAIN] VotingClassifier (XGB + RF) with class_weight='balanced'...")

    with MLflowTracker("FitVision_Deadlift_P0") as tracker:
        xgb_params = dict(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1)
        )
        rf_params = dict(n_estimators=150, max_depth=15, min_samples_leaf=3)
        
        tracker.log_params({"xgb_" + k: v for k, v in xgb_params.items()})
        tracker.log_params({"rf_" + k: v for k, v in rf_params.items()})
        
        m_xgb = xgb.XGBClassifier(**xgb_params, eval_metric='logloss', random_state=42, n_jobs=-1)
        m_rf = RandomForestClassifier(**rf_params, class_weight='balanced', random_state=42, n_jobs=-1)
    
        ensemble = VotingClassifier(
            estimators=[('xgb', m_xgb), ('rf', m_rf)], voting='soft')
    
        ensemble.fit(X_res, y_res)
    
        # ── Evaluate ──
        y_pred = ensemble.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        cm = confusion_matrix(y_test, y_pred)
        
        tracker.log_metrics({"accuracy": acc, "f1_macro": f1})
    
        print(f"\n[RESULT] Accuracy: {acc:.4f}  F1-macro: {f1:.4f}")
        print(f"[RESULT] Confusion matrix:\n{cm}")
        print(classification_report(y_test, y_pred, target_names=['Incorrect', 'Correct']))
    
        # ── Save ──
        model_path = MODELS_DIR / 'deadlift_form.pkl'
        backup_path = MODELS_DIR / 'deadlift_form_pre_p0.pkl'
        if model_path.exists() and not backup_path.exists():
            import shutil
            shutil.copy2(model_path, backup_path)
            print(f"[BACKUP] Old model → {backup_path.name}")
    
        joblib.dump({
            'model': ensemble,
            'feature_cols': FEATURE_COLS,
            'model_type': 'binary_p0_fixed',
            'accuracy': acc,
            'f1_macro': f1,
            'confusion_matrix': cm.tolist(),
            'fixes': [
                'P0-4: class_weight balanced + SMOTE',
                'elbow/knee_symmetry recomputed from angles',
                'GroupShuffleSplit(video_name)',
            ],
            'train_videos': list(train_videos),
            'test_videos': list(test_videos),
        }, model_path)
        
        tracker.log_artifact(str(model_path))

    size_mb = model_path.stat().st_size / 1024 / 1024
    print(f"\n[SAVED] {model_path} ({size_mb:.1f} MB)")

    print(f"\n{'=' * 60}")
    print(f"[SUMMARY] Deadlift P0-4")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  F1-macro:  {f1:.4f}")
    print(f"  Fixes:     imbalance, symmetry bug, group split")
    print(f"  Model:     {model_path.name} ({size_mb:.1f} MB)")
    print(f"  IMPORTANT: Client must be updated to send continuous symmetry")
    print(f"             (see camera/page.tsx:262-263)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
