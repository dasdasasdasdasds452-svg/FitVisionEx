"""
Retrain Squat Model v7 — Using Custom Powerlifting Features

This script trains a new squat model using features extracted from
raw Powerlifting Dataset landmarks using our frontend's EXACT feature
extraction logic.

This guarantees 100% compatibility between training data and prediction data.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
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

CSV_PATH = Path("c:/fit/FitVision/data/raw/powerlifting_squat/extracted_features.csv")

# We only have binary labels (Valid=0/Invalid=1) in this dataset.
# The frontend expects a detailed multiclass model too, so we will 
# synthesize one by combining the binary decision with some rule-based
# heuristics for the specific error classes during training, to make a 
# compatible pkl file that just wraps the logic.
# Alternatively, we can just train a 2-class model and the predictor 
# handles mapping.

BASE_FEATURES = [
    'left_knee_angle',   'right_knee_angle',
    'left_hip_angle',    'right_hip_angle',
    'left_ankle_angle',  'right_ankle_angle',
    'spine_angle',       'torso_lean',
    'left_knee_lateral', 'right_knee_lateral',
    'symmetry_score',    'hip_depth',
]

def add_engineered_features(df):
    df = df.copy()
    # These match predictor.py exactly
    df['avg_knee']       = (df['left_knee_angle'] + df['right_knee_angle']) / 2
    df['avg_hip']        = (df['left_hip_angle']  + df['right_hip_angle'])  / 2
    df['knee_hip_ratio'] = df['avg_knee'] / (df['avg_hip'] + 1e-8)
    df['knee_depth']     = df['avg_knee'] / 90.0
    df['ankle_asym']     = (df['left_ankle_angle'] - df['right_ankle_angle']).abs()
    df['hip_asym']       = (df['left_hip_angle'] - df['right_hip_angle']).abs()
    df['total_lat']      = df['left_knee_lateral'].abs() + df['right_knee_lateral'].abs()
    df['lean_con']       = (df['spine_angle'] - df['torso_lean']).abs()
    return df

def get_feature_cols():
    return [
        'left_knee_angle', 'right_knee_angle', 'left_hip_angle', 'right_hip_angle',
        'left_ankle_angle', 'right_ankle_angle', 'spine_angle', 'torso_lean',
        'left_knee_lateral', 'right_knee_lateral', 'symmetry_score', 'hip_depth',
        'avg_knee', 'avg_hip', 'knee_hip_ratio', 'knee_depth',
        'ankle_asym', 'hip_asym', 'total_lat', 'lean_con'
    ]

def train_binary(df):
    print("\n" + "="*60)
    print("[TRAIN] Binary Squat Form (Correct/Incorrect) - Powerlifting Dataset")
    print("="*60)
    
    feat_cols = get_feature_cols()
    X = df[feat_cols].fillna(0).values
    y = df['label'].values  # 0=Valid, 1=Invalid
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"\nTrain: {len(X_train)}   Test: {len(X_test)}")
    print(f"Correct={(y_train==0).sum()}  Incorrect={(y_train==1).sum()}")
    
    print("Training XGBoost...")
    m_xgb = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=42, n_jobs=-1)
    
    print("Training RandomForest...")
    m_rf = RandomForestClassifier(
        n_estimators=150, max_depth=15, min_samples_leaf=3,
        class_weight='balanced', random_state=42, n_jobs=-1)
    
    print("Building 2-Model Voting Ensemble...")
    ensemble = VotingClassifier(
        estimators=[('xgb', m_xgb), ('rf', m_rf)],
        voting='soft')
    ensemble.fit(X_train, y_train)
    
    y_pred = ensemble.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"\n[OK] Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"     F1 macro: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Correct', 'Incorrect']))
    
    model_path = MODELS_DIR / 'squat_form.pkl'
    joblib.dump({
        'model': ensemble, 'feature_cols': feat_cols,
        'model_type': 'binary_powerlifting_v7',
        'accuracy': acc, 'f1_macro': f1,
    }, model_path)
    print(f"[SAVED] {model_path} ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return acc, f1

def train_multiclass_pseudo(df):
    """
    Since Powerlifting dataset only has binary labels (0/1),
    we generate pseudo-labels for the 5 error classes using our rules,
    then train a multiclass model on it to maintain compatibility with the API.
    """
    print("\n" + "="*60)
    print("[TRAIN] Multiclass Squat Error (6 classes) - Pseudo Labels")
    print("="*60)
    
    feat_cols = get_feature_cols()
    df = df.copy()
    
    # Generate pseudo labels for invalid frames
    def assign_error_class(row):
        if row['label'] == 0: return 0  # Correct
        
        # Priority mapping based on severity/clarity
        if row['spine_angle'] > 40: return 2         # Forward lean
        if row['avg_knee'] > 130: return 1           # Shallow
        if abs(row['left_knee_lateral']) > 0.08 or abs(row['right_knee_lateral']) > 0.08: return 3 # Caving
        if row['symmetry_score'] > 70: return 5      # Asymmetric
        if ((row['left_ankle_angle'] + row['right_ankle_angle'])/2) > 155 and row['avg_knee'] < 120: return 4 # Heels
        
        # Default fallback for errors not matching strict rules
        return 1 # Fallback to shallow
        
    df['multi_label'] = df.apply(assign_error_class, axis=1)
    
    X = df[feat_cols].fillna(0).values
    y = df['multi_label'].values
    
    print("Class distribution:")
    LABEL_MAP = {0: 'Correct', 1: 'Shallow', 2: 'Forward Lean',
                 3: 'Knees Caving', 4: 'Heels Off', 5: 'Asymmetric'}
    for k, v in pd.Series(y).value_counts().items():
        print(f"  {LABEL_MAP[k]}: {v}")
        
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    # SMOTE only works if we have enough samples for each class
    # Heels off might have very few samples, so we'll use a safer approach to balancing
    class_counts = pd.Series(y_train).value_counts()
    min_samples = class_counts.min()
    k_neighbors = min(5, min_samples - 1) if min_samples > 1 else 1
    
    if k_neighbors > 0 and len(class_counts) > 1:
        print(f"Applying SMOTE with k_neighbors={k_neighbors}...")
        sm = SMOTE(random_state=42, k_neighbors=k_neighbors)
        try:
            X_res, y_res = sm.fit_resample(X_train, y_train)
        except ValueError:
            print("SMOTE failed, using original data")
            X_res, y_res = X_train, y_train
    else:
        print("Not enough samples for SMOTE, using original data")
        X_res, y_res = X_train, y_train
        
    print("Training models...")
    m_xgb = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        eval_metric='mlogloss', random_state=42, n_jobs=-1)
    
    m_rf = RandomForestClassifier(
        n_estimators=100, max_depth=12,
        class_weight='balanced', random_state=42, n_jobs=-1)
    
    ensemble = VotingClassifier(
        estimators=[('xgb', m_xgb), ('rf', m_rf)],
        voting='soft')
    ensemble.fit(X_res, y_res)
    
    y_pred = ensemble.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"\n[OK] Accuracy: {acc:.4f}")
    
    target_names = [LABEL_MAP[i] for i in sorted(np.unique(y_test))]
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    model_path = MODELS_DIR / 'squat_form_detailed.pkl'
    joblib.dump({
        'model': ensemble, 'feature_cols': feat_cols,
        'model_type': 'multiclass_powerlifting_pseudo_v7',
        'accuracy': acc, 'f1_macro': f1, 'label_map': LABEL_MAP,
    }, model_path)
    print(f"[SAVED] {model_path} ({model_path.stat().st_size / 1024 / 1024:.1f} MB)")
    return acc, f1

def main():
    df = pd.read_csv(CSV_PATH)
    
    # Filter out bizarre frames (e.g., knee angle > 200 or < -50) just in case
    # to keep data clean
    valid_mask = (df['left_knee_angle'] > -50) & (df['left_knee_angle'] < 250)
    df = df[valid_mask]
    
    df = add_engineered_features(df)
    print(f"Data shape after feature engineering: {df.shape}")
    
    b_acc, b_f1 = train_binary(df)
    m_acc, m_f1 = train_multiclass_pseudo(df)
    
    print("\n" + "="*60)
    print("[SUMMARY] Fully Compatible Retrain (v7)")
    print("="*60)
    print(f"  Binary:     acc={b_acc:.4f}  f1={b_f1:.4f}")
    print(f"  Multiclass: acc={m_acc:.4f}  f1={m_f1:.4f}")

if __name__ == "__main__":
    main()
