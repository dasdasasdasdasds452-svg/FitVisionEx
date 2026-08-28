import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

def eval_metrics(y_true, y_pred):
    avg = 'macro'
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
    rec = recall_score(y_true, y_pred, average=avg, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)
    return acc, prec, rec, f1

print("--- Calculating Poster Comparison Metrics ---")

# 2. DEADLIFT
dl_csv = "C:/fit/FitVision/data/processed/training_dataset.csv"
df = pd.read_csv(dl_csv, low_memory=False)
df_dl = df[df['exercise'] == 'deadlift']
exclude = ['frame', 'timestamp', 'exercise', 'form_correct', 'risk_level', 'deadlift_type', 'video_name', 'score', 'error_type']
feat_cols = [c for c in df_dl.columns if c not in exclude]
X = df_dl[feat_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
y = df_dl['form_correct'].astype(int).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
acc, prec, rec, f1 = eval_metrics(y_test, dt.predict(X_test))
print(f"Deadlift DT (Base): Acc={acc*100:.2f} Prec={prec*100:.2f} Rec={rec*100:.2f} F1={f1*100:.2f}")

# 3. BENCH PRESS
bp_csv = "C:/fit/FitVision/data/interim/benchpress_features.csv"
df_bp = pd.read_csv(bp_csv)
exclude_bp = ['frame', 'timestamp', 'video_name', 'label']
feat_bp = [c for c in df_bp.columns if c not in exclude_bp]
X_bp = df_bp[feat_bp].fillna(0).values
y_bp = df_bp['label'].values
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X_bp, y_bp, df_bp['video_name'].values))
X_train_bp, X_test_bp = X_bp[train_idx], X_bp[test_idx]
y_train_bp, y_test_bp = y_bp[train_idx], y_bp[test_idx]
X_res_bp, y_res_bp = sm.fit_resample(X_train_bp, y_train_bp)

dt_bp = DecisionTreeClassifier(random_state=42)
dt_bp.fit(X_res_bp, y_res_bp)
acc, prec, rec, f1 = eval_metrics(y_test_bp, dt_bp.predict(X_test_bp))
print(f"BenchPress DT (Base): Acc={acc*100:.2f} Prec={prec*100:.2f} Rec={rec*100:.2f} F1={f1*100:.2f}")
