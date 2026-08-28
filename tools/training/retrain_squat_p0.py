"""
Retrain Squat Model — P0 Fix

Fixes TWO critical issues from Production Assessment:
  P0-2: Training/serving skew — force torso_lean = spine_angle (matches client)
  P0-3: Data leakage — use GroupShuffleSplit by video_file instead of random split

Produces: squat_form.pkl (binary only — multiclass replaced by rule-based in predictor.py)
"""
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import sys

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

# ── Paths (relative to script location) ──
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODELS_DIR = PROJECT_ROOT / "data" / "models"
KAGGLE_CSV = PROJECT_ROOT / "data" / "raw" / "kaggle_squat" / "squat_features_augmented.csv"

BASE_FEATURES = [
    'left_knee_angle',   'right_knee_angle',
    'left_hip_angle',    'right_hip_angle',
    'left_ankle_angle',  'right_ankle_angle',
    'spine_angle',       'torso_lean',
    'left_knee_lateral', 'right_knee_lateral',
    'symmetry_score',    'hip_depth',
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features matching predictor.py's engineer_squat_features()."""
    df = df.copy()
    df['avg_knee_angle'] = (df['left_knee_angle'] + df['right_knee_angle']) / 2
    df['avg_hip_angle']  = (df['left_hip_angle']  + df['right_hip_angle'])  / 2
    df['knee_hip_ratio'] = df['avg_knee_angle'] / (df['avg_hip_angle'] + 1e-8)
    df['knee_depth_ratio'] = df['avg_knee_angle'] / 90.0
    df['ankle_asymmetry'] = (df['left_ankle_angle'] - df['right_ankle_angle']).abs()
    df['hip_asymmetry'] = (df['left_hip_angle'] - df['right_hip_angle']).abs()
    df['total_lateral'] = df['left_knee_lateral'].abs() + df['right_knee_lateral'].abs()
    df['lean_consistency'] = (df['spine_angle'] - df['torso_lean']).abs()
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    eng_cols = [
        'avg_knee_angle', 'avg_hip_angle', 'knee_hip_ratio',
        'knee_depth_ratio', 'ankle_asymmetry', 'hip_asymmetry',
        'total_lateral', 'lean_consistency',
    ]
    return BASE_FEATURES + [c for c in eng_cols if c in df.columns]


def main():
    print("=" * 60)
    print("[P0] Retrain Squat — Fix skew + Fix leakage")
    print("=" * 60)

    if not KAGGLE_CSV.exists():
        print(f"[ERROR] CSV not found: {KAGGLE_CSV}")
        return

    df = pd.read_csv(KAGGLE_CSV)
    print(f"[OK] Loaded {len(df)} rows from {KAGGLE_CSV.name}")
    print(f"     video_file unique: {df['video_file'].nunique()}")
    print(f"     Label distribution:\n{df['label'].value_counts().sort_index().to_string()}")

    # ── P0-2 FIX: Force torso_lean = spine_angle ──
    # Client sends torso_lean = spine_angle (identical).
    # Original Kaggle data had them as different values → distribution shift.
    # By forcing them equal in training, train/serve distributions match.
    print(f"\n[P0-2] Before fix: torso_lean mean={df['torso_lean'].mean():.2f}, "
          f"spine_angle mean={df['spine_angle'].mean():.2f}")
    df['torso_lean'] = df['spine_angle']
    print(f"[P0-2] After fix:  torso_lean = spine_angle (lean_consistency will be 0)")

    # Engineer features AFTER the fix
    df = add_engineered_features(df)
    feat_cols = get_feature_cols(df)
    print(f"\n[OK] {len(feat_cols)} features: {feat_cols}")

    # Verify lean_consistency is 0
    assert df['lean_consistency'].sum() == 0, "lean_consistency should be 0 after fix"
    print("[OK] lean_consistency verified = 0 (matches serving behavior)")

    X = df[feat_cols].fillna(0).values
    groups = df['video_file'].values

    # ── Binary classification ──
    print("\n" + "=" * 60)
    print("[TRAIN] Binary — Correct (0) vs Incorrect (1-5)")
    print("=" * 60)

    y_bin = (df['label'] != 0).astype(int)
    print(f"Binary dist: Correct={int((y_bin == 0).sum())}, Incorrect={int((y_bin == 1).sum())}")

    # ── P0-3 FIX: GroupShuffleSplit by video_file ──
    # Random split leaks correlated frames from same video across train/test.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y_bin, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_bin.values[train_idx], y_bin.values[test_idx]

    train_videos = set(groups[train_idx])
    test_videos = set(groups[test_idx])
    print(f"\n[P0-3] GroupShuffleSplit by video_file:")
    print(f"  Train: {len(X_train)} samples, {len(train_videos)} videos")
    print(f"  Test:  {len(X_test)} samples, {len(test_videos)} videos")
    print(f"  Overlap: {train_videos & test_videos}  (should be empty)")

    # SMOTE on training data only
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"  After SMOTE: {len(X_res)} samples")

    # Train ensemble
    m_xgb = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=42, n_jobs=-1)
    m_rf = RandomForestClassifier(
        n_estimators=150, max_depth=15, min_samples_leaf=3,
        class_weight='balanced', random_state=42, n_jobs=-1)

    ensemble = VotingClassifier(
        estimators=[('xgb', m_xgb), ('rf', m_rf)], voting='soft')

    print("\n[TRAINING] VotingClassifier (XGB + RF)...")
    ensemble.fit(X_res, y_res)

    # Evaluate
    y_pred = ensemble.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n[RESULT] Accuracy: {acc:.4f}  F1-macro: {f1:.4f}")
    print(f"[RESULT] Confusion matrix:\n{cm}")
    print(classification_report(y_test, y_pred, target_names=['Correct', 'Incorrect']))

    # ── Save model ──
    model_path = MODELS_DIR / 'squat_form.pkl'

    # Backup old model
    backup_path = MODELS_DIR / 'squat_form_pre_p0.pkl'
    if model_path.exists() and not backup_path.exists():
        import shutil
        shutil.copy2(model_path, backup_path)
        print(f"[BACKUP] Old model saved to {backup_path.name}")

    joblib.dump({
        'model': ensemble,
        'feature_cols': feat_cols,
        'model_type': 'binary_p0_fixed',
        'accuracy': acc,
        'f1_macro': f1,
        'fixes': ['P0-2: torso_lean=spine_angle', 'P0-3: GroupShuffleSplit(video_file)'],
        'train_videos': list(train_videos),
        'test_videos': list(test_videos),
        'confusion_matrix': cm.tolist(),
    }, model_path)

    size_mb = model_path.stat().st_size / 1024 / 1024
    print(f"\n[SAVED] {model_path} ({size_mb:.1f} MB)")

    # ── Save updated feature stats ──
    stats = {}
    for c in feat_cols:
        col_data = df[c]
        stats[c] = {
            'mean': float(col_data.mean()),
            'std': float(col_data.std()),
            'min': float(col_data.min()),
            'max': float(col_data.max()),
        }
    stats_path = MODELS_DIR / 'squat_feature_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"[SAVED] Feature stats: {stats_path.name}")

    print(f"\n{'=' * 60}")
    print(f"[SUMMARY]")
    print(f"  Binary accuracy: {acc:.4f} (with group split — HONEST number)")
    print(f"  F1 macro:        {f1:.4f}")
    print(f"  Fixes applied:   P0-2 (skew), P0-3 (leakage)")
    print(f"  Model saved:     {model_path.name}")
    print(f"  Old model:       {backup_path.name}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
