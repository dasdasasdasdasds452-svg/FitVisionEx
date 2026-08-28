import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def get_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=100, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=15),
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=50, max_depth=6, eval_metric='logloss', random_state=42, n_jobs=-1),
        'LightGBM': lgb.LGBMClassifier(n_estimators=50, max_depth=6, random_state=42, n_jobs=-1, verbose=-1),
        'CatBoost': CatBoostClassifier(iterations=50, depth=6, verbose=0, random_state=42),
        'Multi-Layer Perceptron (NN)': MLPClassifier(hidden_layer_sizes=(64,), max_iter=100, random_state=42),
    }

def eval_metrics(y_true, y_pred):
    avg = 'macro'
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=avg, zero_division=0)
    rec = recall_score(y_true, y_pred, average=avg, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=avg, zero_division=0)
    return acc, prec, rec, f1

sm = SMOTE(random_state=42)
scaler = StandardScaler()

print("--- EVALUATING DEADLIFT ---")
dl_csv = "C:/fit/FitVision/data/processed/training_dataset.csv"
df = pd.read_csv(dl_csv, low_memory=False)
df_dl = df[df['exercise'] == 'deadlift']
# Subsample for speed
df_dl = df_dl.sample(n=30000, random_state=42)
X = df_dl.drop(columns=['frame', 'timestamp', 'exercise', 'form_correct', 'risk_level', 'deadlift_type', 'video_name', 'score', 'error_type'], errors='ignore').apply(pd.to_numeric, errors='coerce').fillna(0).values
y = df_dl['form_correct'].astype(int).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_res, y_res = sm.fit_resample(X_train, y_train)
X_res_s = scaler.fit_transform(X_res)
X_test_s = scaler.transform(X_test)

models = get_models()
for name, m in models.items():
    if name in ['Logistic Regression', 'Multi-Layer Perceptron (NN)', 'K-Nearest Neighbors']:
        m.fit(X_res_s, y_res)
        a, p, r, f = eval_metrics(y_test, m.predict(X_test_s))
    else:
        m.fit(X_res, y_res)
        a, p, r, f = eval_metrics(y_test, m.predict(X_test))
    print(f"DL|{name}|{a*100:.2f}|{p*100:.2f}|{r*100:.2f}|{f*100:.2f}")

print("--- EVALUATING BENCH PRESS ---")
bp_csv = "C:/fit/FitVision/data/interim/benchpress_features.csv"
df_bp = pd.read_csv(bp_csv)
df_bp = df_bp.sample(n=30000, random_state=42)
X_bp = df_bp.drop(columns=['frame', 'timestamp', 'video_name', 'label'], errors='ignore').fillna(0).values
y_bp = df_bp['label'].values
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X_bp, y_bp, df_bp['video_name'].values))
X_train_bp, X_test_bp = X_bp[train_idx], X_bp[test_idx]
y_train_bp, y_test_bp = y_bp[train_idx], y_bp[test_idx]
X_res_bp, y_res_bp = sm.fit_resample(X_train_bp, y_train_bp)
X_res_bp_s = scaler.fit_transform(X_res_bp)
X_test_bp_s = scaler.transform(X_test_bp)

for name, m in models.items():
    if name in ['Logistic Regression', 'Multi-Layer Perceptron (NN)', 'K-Nearest Neighbors']:
        m.fit(X_res_bp_s, y_res_bp)
        a, p, r, f = eval_metrics(y_test_bp, m.predict(X_test_bp_s))
    else:
        m.fit(X_res_bp, y_res_bp)
        a, p, r, f = eval_metrics(y_test_bp, m.predict(X_test_bp))
    print(f"BP|{name}|{a*100:.2f}|{p*100:.2f}|{r*100:.2f}|{f*100:.2f}")
