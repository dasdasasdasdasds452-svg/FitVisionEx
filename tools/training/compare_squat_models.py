import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

try:
    import lightgbm as lgb
except ImportError:
    import subprocess
    print("Installing LightGBM...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm"])
    import lightgbm as lgb
    
try:
    import catboost as cb
except ImportError:
    import subprocess
    print("Installing CatBoost...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost"])
    import catboost as cb

CSV_PATH = Path("c:/fit/FitVision/data/raw/powerlifting_squat/extracted_features.csv")

def add_engineered_features(df):
    df = df.copy()
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

def evaluate_models():
    print("Loading data...")
    df = pd.read_csv(CSV_PATH)
    valid_mask = (df['left_knee_angle'] > -50) & (df['left_knee_angle'] < 250)
    df = df[valid_mask]
    df = add_engineered_features(df)
    
    feat_cols = get_feature_cols()
    X = df[feat_cols].fillna(0).values
    y = df['label'].values  # Binary: 0=Correct, 1=Incorrect
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
        
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
        "XGBoost": xgb.XGBClassifier(n_estimators=200, max_depth=6, eval_metric='logloss', random_state=42, n_jobs=-1),
        "LightGBM": lgb.LGBMClassifier(n_estimators=200, random_state=42, n_jobs=-1, verbose=-1),
        "CatBoost": cb.CatBoostClassifier(iterations=200, random_state=42, verbose=0, thread_count=-1),
        "Multi-Layer Perceptron (NN)": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    # Also evaluate the ensemble we used
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb.XGBClassifier(n_estimators=200, max_depth=6, eval_metric='logloss', random_state=42, n_jobs=-1)),
            ('rf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
        ],
        voting='soft'
    )
    models["Voting Ensemble (Our Final Model)"] = ensemble
    
    results = []
    
    print("\nTraining and evaluating models...\n")
    print(f"{'Model Name':<35} | {'Accuracy':<10} | {'F1 Score':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 85)
    
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            prec = precision_score(y_test, y_pred, average='macro')
            rec = recall_score(y_test, y_pred, average='macro')
            
            results.append({'Model': name, 'Accuracy': acc, 'F1 Score': f1, 'Precision': prec, 'Recall': rec})
            
            print(f"{name:<35} | {acc * 100:>7.2f}%   | {f1 * 100:>7.2f}%   | {prec * 100:>7.2f}%   | {rec * 100:>7.2f}%")
        except Exception as e:
            print(f"{name:<35} | ERROR: {e}")
            
    print("\nDone!")
    
if __name__ == "__main__":
    evaluate_models()
