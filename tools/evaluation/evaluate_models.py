"""
FitVision — Comprehensive Model Evaluation

Reconstructs the SAME test set used during training (from model metadata)
and computes full metrics: per-class P/R/F1, confusion matrix heatmap,
calibration curve, ROC curve.

Output:
  - data/evaluation/results.json     (machine-readable)
  - data/evaluation/*.png            (visualizations)
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # no GUI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Paths ──
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODELS_DIR = PROJECT_ROOT / "data" / "models"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_confusion_heatmap(cm, class_names, title, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=12, fontweight="bold")
    fig.colorbar(im)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    # Numeric labels in cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_calibration_curve(y_true, y_prob, title, path, n_bins=10):
    """Reliability diagram for binary classifier."""
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    brier = brier_score_loss(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "s-", color="#39FF14", label=f"Model (Brier={brier:.4f})")
    ax.hist(y_prob, range=(0, 1), bins=n_bins, alpha=0.5, color="steelblue", label="Prediction distribution")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper left")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return brier


def save_roc_curve(y_true, y_prob, title, path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#39FF14", lw=2, label=f"ROC curve (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return auc


# ──────────────────────────────────────────────────────────────────────────────
# Per-model evaluators
# ──────────────────────────────────────────────────────────────────────────────

def eval_squat_binary(model_data, df_all):
    """Squat binary: load from squat_real_labels.csv, filter to test split."""
    df_test = df_all[df_all["split"] == "test"].copy()
    feat_cols = model_data["feature_cols"]
    X = df_test[feat_cols].fillna(0).values
    y_true = df_test["binary_label"].values

    model = model_data["model"]
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]  # P(Bad=1)

    cm = confusion_matrix(y_true, y_pred)
    class_names = ["Good", "Bad"]
    save_confusion_heatmap(cm, class_names, "Squat (Binary) — Confusion Matrix",
                           OUTPUT_DIR / "squat_binary_confusion.png")

    brier = save_calibration_curve(y_true, y_prob, "Squat (Binary) — Calibration",
                                   OUTPUT_DIR / "squat_binary_calibration.png")
    auc = save_roc_curve(y_true, y_prob, "Squat (Binary) — ROC",
                         OUTPUT_DIR / "squat_binary_roc.png")

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1])

    return {
        "n_test": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "brier_score": float(brier),
        "roc_auc": float(auc),
        "per_class": {
            "Good": {"precision": float(p[0]), "recall": float(r[0]), "f1": float(f1[0])},
            "Bad":  {"precision": float(p[1]), "recall": float(r[1]), "f1": float(f1[1])},
        },
        "confusion_matrix": cm.tolist(),
        "report": classification_report(y_true, y_pred, target_names=class_names, output_dict=True),
    }


def eval_squat_3class(model_data, df_all):
    """Squat 3-class: load from squat_real_labels.csv, filter to test split."""
    df_test = df_all[df_all["split"] == "test"].copy()
    feat_cols = model_data["feature_cols"]
    X = df_test[feat_cols].fillna(0).values
    y_true = df_test["error_code"].values

    model = model_data["model"]
    y_pred = model.predict(X)

    # Labels are 0/2/4 (matching training). Classes present in test:
    present_labels = sorted(np.unique(np.concatenate([y_true, y_pred])))
    label_to_name = {0: "Good", 2: "Bad Back", 4: "Bad Heel"}
    class_names = [label_to_name[l] for l in present_labels]

    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    save_confusion_heatmap(cm, class_names, "Squat (3-Class) — Confusion Matrix",
                           OUTPUT_DIR / "squat_3class_confusion.png")

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=present_labels, average=None)

    return {
        "n_test": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "per_class": {
            label_to_name[l]: {
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i]),
            }
            for i, l in enumerate(present_labels)
        },
        "confusion_matrix": cm.tolist(),
        "labels": [int(l) for l in present_labels],
        "report": classification_report(y_true, y_pred, labels=present_labels,
                                        target_names=class_names, output_dict=True),
    }


def _reconstruct_test_from_groups(df, test_videos, group_col):
    return df[df[group_col].isin(test_videos)].copy()


def _apply_skew_fixes_dl(df):
    """Apply same symmetry recompute as retrain_deadlift_p0.py."""
    df = df.copy()
    df["elbow_symmetry"] = (df["left_elbow_angle"] - df["right_elbow_angle"]).abs()
    df["knee_symmetry"] = (df["left_knee_angle"] - df["right_knee_angle"]).abs()
    return df


def _apply_skew_fixes_bp(df):
    """Apply same symmetry recompute as retrain_benchpress_p0.py."""
    df = df.copy()
    df["elbow_symmetry"] = (df["left_elbow_angle"] - df["right_elbow_angle"]).abs()
    df["knee_symmetry"] = (df["left_knee_angle"] - df["right_knee_angle"]).abs()
    return df


def eval_binary_from_training_csv(model_data, csv_path, group_col, label_col, exercise_filter, skew_fn, title_prefix, file_prefix):
    """Generic evaluator for binary models trained from training_dataset.csv (deadlift/bench)."""
    df = pd.read_csv(csv_path, low_memory=False)
    if exercise_filter:
        df = df[df["exercise"] == exercise_filter].copy()
    df = skew_fn(df)

    test_videos = model_data.get("test_videos", [])
    if not test_videos:
        print(f"  [WARN] No test_videos in metadata for {file_prefix}; using all data")
        df_test = df
    else:
        df_test = _reconstruct_test_from_groups(df, set(test_videos), group_col)

    feat_cols = model_data["feature_cols"]
    X = df_test[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
    y_true = df_test[label_col].astype(int).values

    # Benchpress label is 0/1; deadlift form_correct is True/False → cast to int
    model = model_data["model"]
    y_pred = model.predict(X)

    # Convention: 0 = correct, 1 = incorrect (both models use this after cast)
    positive_idx = list(model.classes_).index(1) if 1 in model.classes_ else 1
    y_prob = model.predict_proba(X)[:, positive_idx]

    cm = confusion_matrix(y_true, y_pred)
    class_names = ["Correct", "Incorrect"]
    save_confusion_heatmap(cm, class_names, f"{title_prefix} — Confusion Matrix",
                           OUTPUT_DIR / f"{file_prefix}_confusion.png")

    brier = save_calibration_curve(y_true, y_prob, f"{title_prefix} — Calibration",
                                   OUTPUT_DIR / f"{file_prefix}_calibration.png")
    auc = save_roc_curve(y_true, y_prob, f"{title_prefix} — ROC",
                         OUTPUT_DIR / f"{file_prefix}_roc.png")

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1])

    return {
        "n_test": len(y_true),
        "n_test_videos": len(set(df_test[group_col].values)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "brier_score": float(brier),
        "roc_auc": float(auc),
        "per_class": {
            "Correct":   {"precision": float(p[0]), "recall": float(r[0]), "f1": float(f1[0])},
            "Incorrect": {"precision": float(p[1]), "recall": float(r[1]), "f1": float(f1[1])},
        },
        "confusion_matrix": cm.tolist(),
        "report": classification_report(y_true, y_pred, target_names=class_names, output_dict=True),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 72)
    print("[EVAL] FitVision — Comprehensive Model Evaluation")
    print("=" * 72)

    results = {}

    # ── Squat: load real-labels CSV once ──
    squat_csv = DATA_DIR / "processed" / "squat_real_labels.csv"
    df_squat = pd.read_csv(squat_csv) if squat_csv.exists() else None

    # ── 1. Squat binary ──
    print("\n[1/5] Squat binary (squat_form.pkl)")
    p = MODELS_DIR / "squat_form.pkl"
    if p.exists() and df_squat is not None:
        m = joblib.load(p)
        results["squat_binary"] = eval_squat_binary(m, df_squat)
        print(f"  acc={results['squat_binary']['accuracy']:.4f}  "
              f"f1={results['squat_binary']['f1_macro']:.4f}  "
              f"brier={results['squat_binary']['brier_score']:.4f}  "
              f"auc={results['squat_binary']['roc_auc']:.4f}")
    else:
        print("  SKIP — model or data not found")

    # ── 2. Squat 3-class ──
    print("\n[2/5] Squat 3-class (squat_form_3class.pkl)")
    p = MODELS_DIR / "squat_form_3class.pkl"
    if p.exists() and df_squat is not None:
        m = joblib.load(p)
        results["squat_3class"] = eval_squat_3class(m, df_squat)
        print(f"  acc={results['squat_3class']['accuracy']:.4f}  "
              f"f1={results['squat_3class']['f1_macro']:.4f}")
        for cls, m_v in results["squat_3class"]["per_class"].items():
            print(f"    {cls}: P={m_v['precision']:.3f} R={m_v['recall']:.3f} F1={m_v['f1']:.3f}")
    else:
        print("  SKIP — model or data not found")

    # ── 3. Deadlift ──
    print("\n[3/5] Deadlift (deadlift_form.pkl)")
    p = MODELS_DIR / "deadlift_form.pkl"
    if p.exists():
        m = joblib.load(p)
        results["deadlift"] = eval_binary_from_training_csv(
            m,
            csv_path=DATA_DIR / "processed" / "training_dataset.csv",
            group_col="video_name",
            label_col="form_correct",
            exercise_filter="deadlift",
            skew_fn=_apply_skew_fixes_dl,
            title_prefix="Deadlift",
            file_prefix="deadlift",
        )
        r = results["deadlift"]
        print(f"  acc={r['accuracy']:.4f}  f1={r['f1_macro']:.4f}  "
              f"brier={r['brier_score']:.4f}  auc={r['roc_auc']:.4f}  "
              f"n_test={r['n_test']} (videos={r['n_test_videos']})")
    else:
        print("  SKIP — model not found")

    # ── 4. Benchpress ──
    print("\n[4/5] Benchpress (benchpress_form.pkl)")
    p = MODELS_DIR / "benchpress_form.pkl"
    bp_csv = DATA_DIR / "interim" / "benchpress_features.csv"
    if p.exists() and bp_csv.exists():
        m = joblib.load(p)
        # Benchpress CSV uses 'label' column directly (0/1), no exercise filter
        results["benchpress"] = eval_binary_from_training_csv(
            m,
            csv_path=bp_csv,
            group_col="video_name",
            label_col="label",
            exercise_filter=None,
            skew_fn=_apply_skew_fixes_bp,
            title_prefix="Benchpress",
            file_prefix="benchpress",
        )
        r = results["benchpress"]
        print(f"  acc={r['accuracy']:.4f}  f1={r['f1_macro']:.4f}  "
              f"brier={r['brier_score']:.4f}  auc={r['roc_auc']:.4f}  "
              f"n_test={r['n_test']} (videos={r['n_test_videos']})")
    else:
        print("  SKIP — model or data not found")

    # ── 5. exercise_classifier (skip — single class likely) ──
    print("\n[5/5] exercise_classifier — SKIP (not the focus of this report)")

    # ── Save JSON ──
    out_json = OUTPUT_DIR / "results.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVED] {out_json}")
    print(f"[SAVED] PNG visualizations in {OUTPUT_DIR}")

    # ── Summary table ──
    print(f"\n{'=' * 72}")
    print("[SUMMARY] All Models")
    print(f"{'=' * 72}")
    print(f"{'Model':<20} {'Acc':>8} {'F1-mac':>8} {'Brier':>8} {'AUC':>8}")
    print("-" * 56)
    for name, r in results.items():
        print(f"{name:<20} {r['accuracy']:>8.4f} {r['f1_macro']:>8.4f} "
              f"{r.get('brier_score', float('nan')):>8.4f} "
              f"{r.get('roc_auc', float('nan')):>8.4f}")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()
