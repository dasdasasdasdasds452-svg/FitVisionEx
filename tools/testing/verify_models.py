"""
Verify the deployed FitVision models through the REAL serving code (app/predictor.py).

Run from the FitVision folder with the same scikit-learn as production (1.3.2):
    C:\\fit\\.conda\\python.exe tools\\testing\\verify_models.py

Checks
  1. Every model loads and returns a response that matches the API schema.
  2. No data leakage: train/test videos do not overlap.
  3. Held-out accuracy using the production thresholds (frame level AND per video).
  4. The serving functions give the same answer as the raw model (no glue bugs).
  5. Client/training feature contract: the distance features the browser sends
     must be computed the same way as in training.

Writes data/evaluation/verify_models_<date>.json and exits 1 if a hard check fails.
"""
from __future__ import annotations

import json
import random
import sys
import time
import warnings
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import logging  # noqa: E402

# Keep per-prediction log lines out of the report
logging.disable(logging.WARNING)
try:
    import structlog

    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.ERROR))
except ImportError:
    # The dev .conda env may not have structlog (production does). predictor.py only
    # needs get_logger(), so stand in with the stdlib logger instead of installing anything.
    import types

    _stub = types.ModuleType("structlog")
    _stub.get_logger = logging.getLogger  # type: ignore[attr-defined]
    sys.modules["structlog"] = _stub

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score  # noqa: E402

from app.predictor import ModelRegistry  # noqa: E402
from app.schemas import FormPrediction  # noqa: E402
from config.settings import MODELS_DIR, PREDICTION_THRESHOLDS  # noqa: E402

DATA = ROOT / "data"
FEATURE_COLS = [
    "left_elbow_angle", "right_elbow_angle",
    "left_shoulder_angle", "right_shoulder_angle",
    "left_hip_angle", "right_hip_angle",
    "left_knee_angle", "right_knee_angle",
    "shoulder_width", "hip_width", "torso_length",
    "elbow_symmetry", "knee_symmetry",
]
SQUAT_RAW = [
    "left_knee_angle", "right_knee_angle", "left_hip_angle", "right_hip_angle",
    "left_ankle_angle", "right_ankle_angle", "spine_angle", "torso_lean",
    "left_knee_lateral", "right_knee_lateral", "symmetry_score", "hip_depth",
]

report: dict = {"date": str(date.today()), "versions": {}, "checks": [], "models": {}}
hard_failures: list[str] = []


def check(name: str, ok: bool, detail: str = "", hard: bool = True) -> None:
    status = "PASS" if ok else ("FAIL" if hard else "WARN")
    print(f"  [{status}] {name}{' — ' + detail if detail else ''}")
    report["checks"].append({"name": name, "status": status, "detail": detail})
    if not ok and hard:
        hard_failures.append(name)


def binary_metrics(y_true_correct: np.ndarray, y_pred_correct: np.ndarray) -> dict:
    """1 = correct form, 0 = incorrect form."""
    cm = confusion_matrix(y_true_correct, y_pred_correct, labels=[1, 0])
    tp_c, fn_c = int(cm[0, 0]), int(cm[0, 1])  # truly correct → predicted correct / incorrect
    fp_c, tn_c = int(cm[1, 0]), int(cm[1, 1])  # truly incorrect → predicted correct / incorrect
    return {
        "n": int(len(y_true_correct)),
        "accuracy": round(float(accuracy_score(y_true_correct, y_pred_correct)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true_correct, y_pred_correct)), 4),
        "f1_macro": round(float(f1_score(y_true_correct, y_pred_correct, average="macro")), 4),
        # What the user feels:
        "false_alarm_rate": round(fn_c / max(1, tp_c + fn_c), 4),  # good rep told "wrong"
        "miss_rate": round(fp_c / max(1, fp_c + tn_c), 4),          # bad rep told "good"
        "confusion[correct,incorrect]": [[tp_c, fn_c], [fp_c, tn_c]],
    }


def per_video(videos: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Majority vote per video — closer to what a user sees over a set."""
    df = pd.DataFrame({"v": videos, "t": y_true, "p": y_pred})
    g = df.groupby("v").agg(t=("t", "mean"), p=("p", "mean"))
    t = (g["t"] >= 0.5).astype(int).values
    p = (g["p"] >= 0.5).astype(int).values
    return {"videos": int(len(g)), "accuracy": round(float(accuracy_score(t, p)), 4),
            "balanced_accuracy": round(float(balanced_accuracy_score(t, p)), 4) if len(set(t)) > 1 else None}


def p_correct(bundle: dict, X: np.ndarray, correct_label: int) -> np.ndarray:
    model = bundle["model"]
    classes = list(model.classes_)
    return model.predict_proba(X)[:, classes.index(correct_label)]


def eval_13feature(name: str, bundle_path: Path, df: pd.DataFrame, label_correct: np.ndarray,
                   correct_label: int, threshold: float, serve_fn) -> None:
    print(f"\n── {name} ({bundle_path.name}) ──")
    bundle = joblib.load(bundle_path)
    meta = bundle
    if not bundle.get("test_videos"):
        # Calibrated wrappers may not copy the split lists — read them from the base model.
        base = bundle_path.with_name(bundle_path.name.replace("_calibrated", ""))
        if base != bundle_path and base.exists():
            meta = joblib.load(base)
    train_v = set(meta.get("train_videos") or [])
    test_v = set(meta.get("test_videos") or [])
    check(f"{name}: train/test video lists stored in model", bool(test_v), f"{len(train_v)} train / {len(test_v)} test videos")
    check(f"{name}: no video in both train and test", not (train_v & test_v), f"overlap={len(train_v & test_v)}")

    df = df.copy()
    df["elbow_symmetry"] = (df["left_elbow_angle"] - df["right_elbow_angle"]).abs()
    df["knee_symmetry"] = (df["left_knee_angle"] - df["right_knee_angle"]).abs()
    mask = df["video_name"].isin(test_v).values
    X = df.loc[mask, FEATURE_COLS].apply(pd.to_numeric, errors="coerce").fillna(0).values
    y = label_correct[mask]
    vids = df.loc[mask, "video_name"].values
    if len(X) == 0:
        check(f"{name}: held-out rows found", False, "no rows for test videos")
        return

    pc = p_correct(bundle, X, correct_label)
    pred = (pc >= threshold).astype(int)
    m = binary_metrics(y, pred)
    m["per_video"] = per_video(vids, y, pred)
    m["threshold"] = threshold
    m["recorded_in_model"] = {k: bundle.get(k) for k in ("accuracy", "f1_macro") if k in bundle}
    report["models"][name] = m
    print(f"  held-out frames: {m['n']}  acc={m['accuracy']}  balanced={m['balanced_accuracy']}  "
          f"false-alarm={m['false_alarm_rate']}  miss={m['miss_rate']}")
    print(f"  per video: {m['per_video']}")
    check(f"{name}: better than chance (balanced acc > 0.60)", m["balanced_accuracy"] > 0.60, str(m["balanced_accuracy"]), hard=False)

    # Serving path must agree with the raw model
    idx = random.Random(0).sample(range(len(X)), min(40, len(X)))
    mism = 0
    for i in idx:
        out = serve_fn(X[i].tolist())
        FormPrediction(**out)
        if bool(out["form_correct"]) != bool(pred[i]):
            mism += 1
    check(f"{name}: serving code == raw model on {len(idx)} rows", mism == 0, f"{mism} mismatches")


def eval_squat(reg: ModelRegistry) -> None:
    print("\n── squat (squat_form_3class.pkl) ──")
    df = pd.read_csv(DATA / "processed" / "squat_real_labels.csv")
    test = df[df["split"] == "test"].reset_index(drop=True)
    check("squat: held-out test split exists", len(test) > 0, f"{len(test)} images")
    bundle = joblib.load(MODELS_DIR / "squat_form_3class.pkl")
    from app.predictor import engineer_squat_features
    X = np.array([engineer_squat_features(r) for r in test[SQUAT_RAW].to_dict("records")])
    y = test["error_code"].astype(int).values
    raw = bundle["model"].predict(X)
    m = {"n": int(len(y)), "raw_3class_accuracy": round(float(accuracy_score(y, raw)), 4),
         "raw_3class_f1_macro": round(float(f1_score(y, raw, average="macro")), 4)}

    # Full serving path (standing check, low-confidence rule override, …)
    served_correct, served_codes = [], []
    for r in test[SQUAT_RAW].to_dict("records"):
        out = reg.predict_squat(r)
        FormPrediction(**out)
        served_correct.append(int(out["form_correct"]))
        served_codes.append(int(out.get("error_code") or 0))
    served_correct = np.array(served_correct)
    m["served_binary"] = binary_metrics((y == 0).astype(int), served_correct)
    standing = int(sum(1 for r in X if r[12] > 150))
    m["rows_skipped_by_standing_check"] = standing
    report["models"]["squat"] = m
    print(f"  raw 3-class acc={m['raw_3class_accuracy']} f1={m['raw_3class_f1_macro']}")
    sb = m["served_binary"]
    print(f"  served good/bad: acc={sb['accuracy']} balanced={sb['balanced_accuracy']} "
          f"false-alarm={sb['false_alarm_rate']} miss={sb['miss_rate']}  (standing-check rows: {standing})")
    check("squat: better than chance (balanced acc > 0.60)", sb["balanced_accuracy"] > 0.60, str(sb["balanced_accuracy"]), hard=False)


def contract_check() -> None:
    """The browser must compute the 3 distance features like extract_features.py does."""
    print("\n── client ↔ training feature contract ──")
    page = ROOT.parent / "fitvision-next" / "src" / "app" / "camera" / "page.tsx"
    if not page.exists():
        check("client source found", False, str(page), hard=False)
        return
    src = page.read_text(encoding="utf-8")
    uses_dist = "dist2d(lm[11], lm[12])" in src and "dist2d(lm[23], lm[24])" in src and "dist2d(lm[11], lm[23])" in src
    check("client shoulder/hip width + torso length match training (2D distance, left side torso)", uses_dist,
          "camera/page.tsx must use dist2d(...) like extract_features.py")


def main() -> int:
    t0 = time.time()
    warnings.filterwarnings("ignore", category=UserWarning)
    report["versions"] = {"python": sys.version.split()[0], "sklearn": sklearn.__version__, "numpy": np.__version__}
    print(f"Python {report['versions']['python']}  scikit-learn {sklearn.__version__}  numpy {np.__version__}")
    check("scikit-learn matches production (1.3.2)", sklearn.__version__ == "1.3.2", sklearn.__version__, hard=False)

    reg = ModelRegistry(MODELS_DIR)

    print("\n── smoke: each endpoint returns a valid response ──")
    sample13 = [160, 160, 40, 40, 170, 170, 175, 175, 0.21, 0.14, 0.18, 0, 0]
    for name, fn in (("deadlift", reg.predict_deadlift), ("benchpress", reg.predict_benchpress)):
        try:
            out = fn(sample13)
            FormPrediction(**out)
            check(f"{name}: loads and returns schema-valid output", out.get("feedback") != "Model not loaded", out.get("feedback", ""))
        except Exception as e:  # noqa: BLE001
            check(f"{name}: loads and returns schema-valid output", False, repr(e))
    standing = dict(zip(SQUAT_RAW, [175, 175, 175, 175, 100, 100, 5, 5, 0, 0, 0, 0.5]))
    deep_valgus = dict(zip(SQUAT_RAW, [80, 82, 70, 72, 70, 70, 20, 20, 0.12, 0.11, 10, 0.7]))
    try:
        s1 = reg.predict_squat(standing); FormPrediction(**s1)
        s2 = reg.predict_squat(deep_valgus); FormPrediction(**s2)
        check("squat: standing pose is not flagged", s1["form_correct"] is True)
        check("squat: deep squat with knees caving gets a risk assessment", s2.get("risk_assessment") is not None,
              f"form_correct={s2['form_correct']} error={s2.get('error_type')}")
    except Exception as e:  # noqa: BLE001
        check("squat: model loads and predicts", False, repr(e))

    td = pd.read_csv(DATA / "processed" / "training_dataset.csv", low_memory=False)
    dl = td[td["exercise"] == "deadlift"].reset_index(drop=True)
    cal = MODELS_DIR / "deadlift_form_calibrated.pkl"
    eval_13feature("deadlift", cal if cal.exists() else MODELS_DIR / "deadlift_form.pkl", dl,
                   (dl["form_correct"] == True).astype(int).values, 1,  # noqa: E712
                   PREDICTION_THRESHOLDS.get("deadlift", 0.5), reg.predict_deadlift)

    bp = pd.read_csv(DATA / "interim" / "benchpress_features.csv")
    eval_13feature("benchpress", MODELS_DIR / "benchpress_form.pkl", bp,
                   (bp["label"] == 0).astype(int).values, 0,
                   PREDICTION_THRESHOLDS.get("benchpress", 0.5), reg.predict_benchpress)

    eval_squat(reg)
    contract_check()

    out_path = DATA / "evaluation" / f"verify_models_{date.today()}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport: {out_path}  ({time.time() - t0:.0f}s)")
    if hard_failures:
        print(f"FAILED: {len(hard_failures)} hard check(s): {hard_failures}")
        return 1
    print("ALL HARD CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
