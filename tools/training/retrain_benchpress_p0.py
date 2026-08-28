"""
Retrain Benchpress Model — P0 Fix

Fixes:
  1. elbow_symmetry/knee_symmetry = 0 bug → recompute from angle columns
  2. Data leakage → GroupShuffleSplit by video_name
  3. Proper per-class metrics

Produces: benchpress_form.pkl
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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODELS_DIR = PROJECT_ROOT / "data" / "models"
DATA_CSV = PROJECT_ROOT / "data" / "interim" / "benchpress_features.csv"

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
    print("[P0] Retrain Benchpress — Fix symmetry bug + leakage")
    print("=" * 60)

    if not DATA_CSV.exists():
        print(f"[ERROR] CSV not found: {DATA_CSV}")
        return

    df = pd.read_csv(DATA_CSV)
    print(f"[OK] Loaded {len(df)} benchpress rows")
    print(f"     label: Correct(0)={int((df['label'] == 0).sum())}, "
          f"Incorrect(1)={int((df['label'] == 1).sum())}")
    print(f"     video_name unique: {df['video_name'].nunique()}")

    # ── Fix symmetry bug ──
    print(f"\n[FIX] elbow_symmetry before: all zeros = {(df['elbow_symmetry'] == 0).all()}")
    print(f"[FIX] knee_symmetry before:  all zeros = {(df['knee_symmetry'] == 0).all()}")

    df['elbow_symmetry'] = (df['left_elbow_angle'] - df['right_elbow_angle']).abs()
    df['knee_symmetry'] = (df['left_knee_angle'] - df['right_knee_angle']).abs()

    print(f"[FIX] elbow_symmetry after:  mean={df['elbow_symmetry'].mean():.2f}, "
          f"std={df['elbow_symmetry'].std():.2f}")
    print(f"[FIX] knee_symmetry after:   mean={df['knee_symmetry'].mean():.2f}, "
          f"std={df['knee_symmetry'].std():.2f}")

    X = df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce').fillna(0).values
    y = df['label'].values  # 0=correct, 1=incorrect
    groups = df['video_name'].values

    # ── GroupShuffleSplit ──
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    train_videos = set(groups[train_idx])
    test_videos = set(groups[test_idx])
    print(f"\n[SPLIT] GroupShuffleSplit by video_name:")
    print(f"  Train: {len(X_train)} samples, {len(train_videos)} videos")
    print(f"  Test:  {len(X_test)} samples, {len(test_videos)} videos")
    print(f"  Overlap: {train_videos & test_videos}")
    print(f"  Train labels: {Counter(y_train)}")
    print(f"  Test labels:  {Counter(y_test)}")

    # SMOTE
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"\n[SMOTE] After: {len(X_res)} ({Counter(y_res)})")

    # ── Train ──
    from mlflow_tracker import MLflowTracker
    print("\n[TRAIN] VotingClassifier (XGB + RF)...")
    
    with MLflowTracker("FitVision_Benchpress_P0") as tracker:
        xgb_params = dict(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
        rf_params = dict(n_estimators=150, max_depth=15, min_samples_leaf=3)
        tracker.log_params({"xgb_" + k: v for k, v in xgb_params.items()})
        tracker.log_params({"rf_" + k: v for k, v in rf_params.items()})
        
        m_xgb = xgb.XGBClassifier(**xgb_params, eval_metric='logloss', random_state=42, n_jobs=-1)
        m_rf = RandomForestClassifier(**rf_params, class_weight='balanced', random_state=42, n_jobs=-1)
    
        ensemble = VotingClassifier(
            estimators=[('xgb', m_xgb), ('rf', m_rf)], voting='soft')
        ensemble.fit(X_res, y_res)
    
        y_pred = ensemble.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        cm = confusion_matrix(y_test, y_pred)
        
        tracker.log_metrics({"accuracy": acc, "f1_macro": f1})
    
        print(f"\n[RESULT] Accuracy: {acc:.4f}  F1-macro: {f1:.4f}")
        print(f"[RESULT] Confusion matrix:\n{cm}")
        print(classification_report(y_test, y_pred, target_names=['Correct', 'Incorrect']))
    
        # ── Save ──
        model_path = MODELS_DIR / 'benchpress_form.pkl'
        backup_path = MODELS_DIR / 'benchpress_form_pre_p0.pkl'
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
            'fixes': ['symmetry bug fixed', 'GroupShuffleSplit(video_name)'],
            'train_videos': list(train_videos),
            'test_videos': list(test_videos),
        }, model_path)
        
        tracker.log_artifact(str(model_path))
        
    size_mb = model_path.stat().st_size / 1024 / 1024
    print(f"\n[SAVED] {model_path} ({size_mb:.1f} MB)")
    print(f"\n{'=' * 60}")
    print(f"[SUMMARY] Benchpress")
    print(f"  Accuracy: {acc:.4f}  F1-macro: {f1:.4f}")
    print(f"  Model:    {model_path.name} ({size_mb:.1f} MB)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
