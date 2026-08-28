import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

def eval_model(name, data_path, model_path, test_size=0.2):
    try:
        df = pd.read_csv(data_path)
        X = df.drop('label', axis=1)
        y = df['label']
        
        # Split (use same random state to keep it consistent)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Load model
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        # Handle binary vs multiclass
        avg = 'macro' if len(np.unique(y)) > 2 else 'binary'
        prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
        rec = recall_score(y_test, y_pred, average=avg, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)
        
        print(f"[{name}] Acc:{acc:.4f} Prec:{prec:.4f} Rec:{rec:.4f} F1:{f1:.4f}")
    except Exception as e:
        print(f"[{name}] Failed: {e}")

if __name__ == "__main__":
    print("--- Evaluating All Models ---")
    eval_model("Deadlift", "c:/fit/FitVision/data/extracted_features_deadlift.csv", "c:/fit/FitVision/data/models/deadlift_model.pkl")
    eval_model("BenchPress", "c:/fit/FitVision/data/extracted_features_bench_press.csv", "c:/fit/FitVision/data/models/bench_press_model.pkl")
