import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from imblearn.over_sampling import SMOTE

DATA_CSV = "C:/fit/FitVision/data/processed/training_dataset.csv"
df = pd.read_csv(DATA_CSV, low_memory=False)
df_dl = df[df['exercise'] == 'deadlift'].copy()
exclude_cols = ['frame', 'timestamp', 'exercise', 'form_correct', 'risk_level',
               'deadlift_type', 'video_name', 'score', 'error_type']
feature_cols = [col for col in df_dl.columns if col not in exclude_cols]
X = df_dl[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values
y = df_dl['form_correct'].astype(int).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

rf_clf = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1)
rf_clf.fit(X_train, y_train) # train on imbalanced or what? the output before said After SMOTE. wait, in the script I used X_res but fit is actually `rf_clf.fit(X_train, y_train)` in script 46! Let's check `train_deadlift_experiments.py` line 49. Yes, it was trained on X_train not X_res!

y_pred = rf_clf.predict(X_test)
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')
print(f"Deadlift RF Prec: {prec*100:.2f}% Rec: {rec*100:.2f}%")
