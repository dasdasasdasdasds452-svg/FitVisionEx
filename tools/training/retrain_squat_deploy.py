"""
Retrain Squat Model v6 — Original features + Feature Scaler

Strategy: Train with ALL 20 features (Kaggle data as-is, high accuracy)
but SAVE the Kaggle training statistics so the predictor can transform
frontend features to match Kaggle distribution before prediction.

Transform: z = (frontend - frontend_mean) / frontend_std → kaggle_x = z * kaggle_std + kaggle_mean
"""
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import sys

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import MODELS_DIR

KAGGLE_CSV = Path("C:/fit/FitVision/data/raw/kaggle_squat/squat_features_augmented.csv")

BASE_FEATURES = [
    'left_knee_angle',   'right_knee_angle',
    'left_hip_angle',    'right_hip_angle',
    'left_ankle_angle',  'right_ankle_angle',
    'spine_angle',       'torso_lean',
    'left_knee_lateral', 'right_knee_lateral',
    'symmetry_score',    'hip_depth',
]

LABEL_MAP = {0: 'Correct', 1: 'Shallow', 2: 'Forward Lean',
             3: 'Knees Caving', 4: 'Heels Off', 5: 'Asymmetric'}

def add_engineered_features(df):
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

def get_feature_cols(df):
    eng_cols = [
        'avg_knee_angle', 'avg_hip_angle', 'knee_hip_ratio',
        'knee_depth_ratio', 'ankle_asymmetry', 'hip_asymmetry',
        'total_lateral', 'lean_consistency',
    ]
    return BASE_FEATURES + [c for c in eng_cols if c in df.columns]

def save_feature_stats(df, feat_cols):
    """Save Kaggle training data statistics so predictor can transform features"""
    stats = {}
    for c in feat_cols:
        stats[c] = {
            'mean': float(df[c].mean()),
            'std': float(df[c].std()),
            'min': float(df[c].min()),
            'max': float(df[c].max()),
        }
    
    stats_path = MODELS_DIR / 'squat_feature_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"[SAVED] Feature stats: {stats_path}")
    return stats

def main():
    df = pd.read_csv(KAGGLE_CSV)
    
    # ML Alignment (Gotcha #2)
    # The client sends torso_lean exactly identical to spine_angle
    df['torso_lean'] = df['spine_angle']
    
    df = add_engineered_features(df)
    feat_cols = get_feature_cols(df)
    
    print(f"[OK] Loaded {len(df)} rows, {len(feat_cols)} features")
    
    # Save Kaggle statistics for feature transformation
    stats = save_feature_stats(df, feat_cols)
    print(f"\nKaggle feature statistics:")
    for c in feat_cols:
        s = stats[c]
        print(f"  {c:25s}: mean={s['mean']:8.2f}  std={s['std']:8.2f}  range=[{s['min']:.2f}, {s['max']:.2f}]")
    
    X = df[feat_cols].fillna(0).values
    
    # ── Binary ──
    from mlflow_tracker import MLflowTracker
    print("\n" + "="*60)
    print("[TRAIN] Binary — ALL features, XGBoost + RF")
    print("="*60)
    
    y_bin = (df['label'] != 0).astype(int)
    groups = df['video_file'].values
    
    splitter_bin = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter_bin.split(X, y_bin, groups=groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_bin[train_idx], y_bin[test_idx]
    
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    
    with MLflowTracker("FitVision_Squat_Binary") as tracker:
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
        tracker.log_metrics({"accuracy": acc, "f1_macro": f1})
        
        print(f"\n[OK] Accuracy: {acc:.4f}  F1: {f1:.4f}")
        print(classification_report(y_test, y_pred, target_names=['Correct', 'Incorrect']))
        
        model_path = MODELS_DIR / 'squat_form.pkl'
        joblib.dump({
            'model': ensemble, 'feature_cols': feat_cols,
            'model_type': 'binary_v6_with_scaler',
            'accuracy': acc, 'f1_macro': f1,
        }, model_path)
        tracker.log_artifact(str(model_path))
        print(f"[SAVED] {model_path} ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # ── Multiclass ──
    print("\n" + "="*60)
    print("[TRAIN] Multiclass — ALL features, XGBoost + RF")
    print("="*60)
    
    y_multi = df['label'].values
    
    splitter_multi = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx2, test_idx2 = next(splitter_multi.split(X, y_multi, groups=groups))
    X_train, X_test = X[train_idx2], X[test_idx2]
    y_train, y_test = y_multi[train_idx2], y_multi[test_idx2]
    
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    
    with MLflowTracker("FitVision_Squat_Multiclass") as tracker:
        xgb_params = dict(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
        rf_params = dict(n_estimators=150, max_depth=15, min_samples_leaf=3)
        tracker.log_params({"xgb_" + k: v for k, v in xgb_params.items()})
        tracker.log_params({"rf_" + k: v for k, v in rf_params.items()})
        
        m_xgb2 = xgb.XGBClassifier(**xgb_params, eval_metric='mlogloss', random_state=42, n_jobs=-1)
        m_rf2 = RandomForestClassifier(**rf_params, class_weight='balanced', random_state=42, n_jobs=-1)
        
        ensemble2 = VotingClassifier(
            estimators=[('xgb', m_xgb2), ('rf', m_rf2)], voting='soft')
        ensemble2.fit(X_res, y_res)
        
        y_pred = ensemble2.predict(X_test)
        acc2 = accuracy_score(y_test, y_pred)
        f12 = f1_score(y_test, y_pred, average='macro')
        tracker.log_metrics({"accuracy": acc2, "f1_macro": f12})
        
        print(f"\n[OK] Accuracy: {acc2:.4f}  F1: {f12:.4f}")
        target_names = [LABEL_MAP[i] for i in sorted(LABEL_MAP)]
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        model_path2 = MODELS_DIR / 'squat_form_detailed.pkl'
        joblib.dump({
            'model': ensemble2, 'feature_cols': feat_cols,
            'model_type': 'multiclass_v6_with_scaler',
            'accuracy': acc2, 'f1_macro': f12, 'label_map': LABEL_MAP,
        }, model_path2)
        tracker.log_artifact(str(model_path2))
        print(f"[SAVED] {model_path2} ({model_path2.stat().st_size / 1024 / 1024:.1f} MB)")
    
    print(f"\n[SUMMARY] Binary: {acc:.4f}  Multiclass: {acc2:.4f}")
    print(f"[NEXT] Update predictor.py to load squat_feature_stats.json and transform features")

if __name__ == "__main__":
    main()
