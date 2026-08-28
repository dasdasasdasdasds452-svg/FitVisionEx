"""
FitVision — Model Calibration Script
This script calibrates the predict_proba outputs of our deployed models 
using CalibratedClassifierCV to ensure they represent true probabilities.
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, accuracy_score
from sklearn.model_selection import GroupShuffleSplit
import structlog

logger = structlog.get_logger(__name__)

def calibrate_model(model_path: Path, data_path: Path, feature_cols: list, label_col: str, group_col: str = "video_id"):
    """
    Applies Platt scaling (isotonic or sigmoid) to an existing model using a validation set.
    """
    logger.info("Loading model", path=str(model_path))
    model_data = joblib.load(model_path)
    base_model = model_data["model"]
    
    logger.info("Loading dataset", path=str(data_path))
    df = pd.read_csv(data_path)
    
    # Simple split to get a validation set for calibration
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(df, groups=df[group_col]))
    
    val_df = df.iloc[val_idx]
    X_val = val_df[feature_cols].values
    y_val = val_df[label_col].values
    
    # Evaluate pre-calibration
    pre_probs = base_model.predict_proba(X_val)
    if len(np.unique(y_val)) == 2:
        pre_brier = brier_score_loss(y_val, pre_probs[:, 1])
        logger.info("Pre-calibration", accuracy=accuracy_score(y_val, base_model.predict(X_val)), brier_score=pre_brier)
    
    logger.info("Calibrating model (Isotonic Regression)...")
    calibrated_clf = CalibratedClassifierCV(base_model, cv='prefit', method='isotonic')
    calibrated_clf.fit(X_val, y_val)
    
    # Evaluate post-calibration
    post_probs = calibrated_clf.predict_proba(X_val)
    if len(np.unique(y_val)) == 2:
        post_brier = brier_score_loss(y_val, post_probs[:, 1])
        logger.info("Post-calibration", accuracy=accuracy_score(y_val, calibrated_clf.predict(X_val)), brier_score=post_brier)
    
    # Save calibrated model back
    model_data["model"] = calibrated_clf
    model_data["is_calibrated"] = True
    calibrated_path = model_path.with_name(model_path.stem + "_calibrated.pkl")
    joblib.dump(model_data, calibrated_path)
    logger.info("Saved calibrated model", path=str(calibrated_path))

if __name__ == "__main__":
    import structlog
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()
        ]
    )

    models_dir = Path(__file__).resolve().parents[2] / "data" / "models"
    data_dir = Path(__file__).resolve().parents[2] / "data" / "processed"

    # ── Deadlift calibration ──────────────────────────────────────────────
    dl_model_path = models_dir / "deadlift_form.pkl"
    dl_data_path = data_dir / "training_dataset.csv"

    DEADLIFT_FEATURE_COLS = [
        "left_elbow_angle", "right_elbow_angle",
        "left_shoulder_angle", "right_shoulder_angle",
        "left_hip_angle", "right_hip_angle",
        "left_knee_angle", "right_knee_angle",
        "shoulder_width", "hip_width", "torso_length",
        "elbow_symmetry", "knee_symmetry",
    ]

    if dl_model_path.exists() and dl_data_path.exists():
        logger.info("Starting calibration for DEADLIFT model")

        # Fix symmetry bug (Gotcha #4) before calibrating
        import pandas as pd
        df = pd.read_csv(dl_data_path, low_memory=False)
        df["elbow_symmetry"] = abs(df["left_elbow_angle"] - df["right_elbow_angle"])
        df["knee_symmetry"] = abs(df["left_knee_angle"] - df["right_knee_angle"])

        # Convert form_correct to int label
        df["label"] = df["form_correct"].astype(int)

        # Save fixed version temporarily
        tmp_path = data_dir / "_dl_cal_tmp.csv"
        df.to_csv(tmp_path, index=False)

        calibrate_model(dl_model_path, tmp_path, DEADLIFT_FEATURE_COLS, "label", group_col="video_name")

        tmp_path.unlink(missing_ok=True)
        logger.info("Deadlift calibration complete!")
    else:
        logger.warning("Deadlift model or dataset not found", model=str(dl_model_path), data=str(dl_data_path))

