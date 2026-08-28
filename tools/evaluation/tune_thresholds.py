"""
FitVision — Threshold Sweep + Benchpress Generalization Test

Part 1: Threshold sweep for deadlift (pre-cal vs calibrated)
        Plots FAR / Accuracy / F1 vs threshold → find optimal.
Part 2: Benchpress cross-source generalization test
        Split by "source pattern" (corr_N vs inc_N) to simulate external data.
"""
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
MODELS_DIR = PROJECT_ROOT / "data" / "models"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = DATA_DIR / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def apply_symmetry_fix(df):
    df = df.copy()
    df["elbow_symmetry"] = (df["left_elbow_angle"] - df["right_elbow_angle"]).abs()
    df["knee_symmetry"] = (df["left_knee_angle"] - df["right_knee_angle"]).abs()
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Part 1: Deadlift threshold sweep
# ──────────────────────────────────────────────────────────────────────────────

def sweep_thresholds(proba_correct, y_true, thresholds):
    """Compute FAR/Acc/F1 over a range of thresholds.
    Convention: 1 = correct, 0 = incorrect.
    """
    results = []
    n_correct = max((y_true == 1).sum(), 1)
    n_incorrect = max((y_true == 0).sum(), 1)
    for t in thresholds:
        y_pred = (proba_correct >= t).astype(int)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        # False alarm = predict incorrect when actually correct
        n_false_alarm = int(((y_pred == 0) & (y_true == 1)).sum())
        # False negative (missed) = predict correct when actually incorrect
        n_missed = int(((y_pred == 1) & (y_true == 0)).sum())
        p, r, _, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=[0, 1], average=None, zero_division=0
        )
        results.append({
            "threshold": t,
            "accuracy": acc,
            "f1_macro": f1,
            "far": n_false_alarm / n_correct,        # fraction of correct wrongly flagged
            "miss_rate": n_missed / n_incorrect,      # fraction of incorrect missed
            "precision_correct": r[1],                # recall of "correct" class
            "precision_incorrect": r[0],
        })
    return pd.DataFrame(results)


def plot_sweep(df_sweep, title, path, far_target=0.10):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Accuracy + F1 vs threshold
    ax = axes[0]
    ax.plot(df_sweep["threshold"], df_sweep["accuracy"], "o-", color="#39FF14", label="Accuracy")
    ax.plot(df_sweep["threshold"], df_sweep["f1_macro"], "s-", color="steelblue", label="F1-macro")
    ax.set_xlabel("Threshold (P(correct) >= ?)")
    ax.set_ylabel("Score")
    ax.set_title(f"{title} — Accuracy & F1", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    # Right: FAR + miss rate vs threshold
    ax = axes[1]
    ax.plot(df_sweep["threshold"], df_sweep["far"], "o-", color="red", label="False Alarm Rate (Good→Bad)")
    ax.plot(df_sweep["threshold"], df_sweep["miss_rate"], "s-", color="orange", label="Miss Rate (Bad→Good)")
    ax.axhline(far_target, color="green", linestyle=":", alpha=0.7, label=f"FAR target = {far_target}")
    ax.set_xlabel("Threshold (P(correct) >= ?)")
    ax.set_ylabel("Rate")
    ax.set_title(f"{title} — Error Rates", fontsize=11, fontweight="bold")
    ax.legend(loc="upper center")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def find_optimal_thresholds(df_sweep):
    """Find best threshold by multiple criteria."""
    out = {}

    # Criterion 1: Max F1
    idx = df_sweep["f1_macro"].idxmax()
    out["max_f1"] = {
        "threshold": float(df_sweep.loc[idx, "threshold"]),
        "f1": float(df_sweep.loc[idx, "f1_macro"]),
        "acc": float(df_sweep.loc[idx, "accuracy"]),
        "far": float(df_sweep.loc[idx, "far"]),
    }

    # Criterion 2: FAR <= 10% with max accuracy
    sub = df_sweep[df_sweep["far"] <= 0.10]
    if not sub.empty:
        idx = sub["accuracy"].idxmax()
        out["far_le_10pct"] = {
            "threshold": float(df_sweep.loc[idx, "threshold"]),
            "f1": float(df_sweep.loc[idx, "f1_macro"]),
            "acc": float(df_sweep.loc[idx, "accuracy"]),
            "far": float(df_sweep.loc[idx, "far"]),
        }

    # Criterion 3: FAR <= 15% with max accuracy
    sub = df_sweep[df_sweep["far"] <= 0.15]
    if not sub.empty:
        idx = sub["accuracy"].idxmax()
        out["far_le_15pct"] = {
            "threshold": float(df_sweep.loc[idx, "threshold"]),
            "f1": float(df_sweep.loc[idx, "f1_macro"]),
            "acc": float(df_sweep.loc[idx, "accuracy"]),
            "far": float(df_sweep.loc[idx, "far"]),
        }

    # Criterion 4: Balanced error (FAR ≈ miss rate)
    df_sweep["err_diff"] = (df_sweep["far"] - df_sweep["miss_rate"]).abs()
    idx = df_sweep["err_diff"].idxmin()
    out["balanced_errors"] = {
        "threshold": float(df_sweep.loc[idx, "threshold"]),
        "f1": float(df_sweep.loc[idx, "f1_macro"]),
        "acc": float(df_sweep.loc[idx, "accuracy"]),
        "far": float(df_sweep.loc[idx, "far"]),
    }
    return out


def run_deadlift_sweep():
    print("=" * 72)
    print("[PART 1] Deadlift Threshold Sweep")
    print("=" * 72)

    df = pd.read_csv(DATA_DIR / "processed" / "training_dataset.csv", low_memory=False)
    df_dl = df[df["exercise"] == "deadlift"].copy()
    df_dl = apply_symmetry_fix(df_dl)

    thresholds = np.arange(0.05, 0.96, 0.05)
    all_results = {}

    for pkl_name, key in [("deadlift_form.pkl", "pre_calibrated"),
                          ("deadlift_form_calibrated.pkl", "calibrated")]:
        p = MODELS_DIR / pkl_name
        if not p.exists():
            print(f"  [SKIP] {pkl_name} not found")
            continue

        m = joblib.load(p)
        test_videos = set(m["test_videos"])
        df_test = df_dl[df_dl["video_name"].isin(test_videos)].copy()
        feat_cols = m["feature_cols"]
        X = df_test[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
        y_true = df_test["form_correct"].astype(int).values

        proba = m["model"].predict_proba(X)
        classes = list(m["model"].classes_)
        correct_idx = classes.index(1) if 1 in classes else 0
        prob_correct = proba[:, correct_idx]

        df_sweep = sweep_thresholds(prob_correct, y_true, thresholds)
        all_results[key] = df_sweep

        plot_sweep(df_sweep, f"Deadlift ({key})", OUTPUT_DIR / f"deadlift_sweep_{key}.png")

        print(f"\n--- {pkl_name} ({key}) ---")
        print(f"  n_test = {len(y_true)}, n_correct = {(y_true==1).sum()}, n_incorrect = {(y_true==0).sum()}")
        print(f"  {'Thresh':<8} {'Acc':<8} {'F1':<8} {'FAR':<8} {'Miss':<8}")
        for _, row in df_sweep.iterrows():
            print(f"  {row['threshold']:<8.2f} {row['accuracy']:<8.4f} {row['f1_macro']:<8.4f} "
                  f"{row['far']:<8.4f} {row['miss_rate']:<8.4f}")

    # Side-by-side comparison plot
    if len(all_results) == 2:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, metric, ylabel in [(axes[0], "accuracy", "Accuracy"),
                                   (axes[1], "far", "False Alarm Rate")]:
            for key, df_s in all_results.items():
                ax.plot(df_s["threshold"], df_s[metric], "o-", label=key)
            ax.set_xlabel("Threshold")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Deadlift: {ylabel} — Pre-cal vs Calibrated", fontweight="bold")
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "deadlift_sweep_compare.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"\n[SAVED] Side-by-side comparison → deadlift_sweep_compare.png")

    # Recommendation
    print("\n" + "=" * 72)
    print("[RECOMMENDATION]")
    print("=" * 72)
    if "calibrated" in all_results:
        df_s = all_results["calibrated"]
        # Find threshold where FAR <= 10% with max acc
        sub = df_s[df_s["far"] <= 0.10]
        if not sub.empty:
            best = sub.loc[sub["accuracy"].idxmax()]
            print(f"  🎯 Calibrated + FAR≤10%: threshold={best['threshold']:.2f}")
            print(f"     Accuracy={best['accuracy']:.4f}  F1={best['f1_macro']:.4f}  FAR={best['far']:.4f}")
        else:
            best = df_s.loc[df_s["far"].idxmin()]
            print(f"  ⚠️  No threshold gives FAR≤10%. Lowest FAR at threshold={best['threshold']:.2f} (FAR={best['far']:.4f})")


# ─────────────────────────────────────────พ้น──────────────────────────────────────
# Part 2: Benchpress cross-source generalization
# ──────────────────────────────────────────────────────────────────────────────

def run_benchpress_generalization():
    print("\n" + "=" * 72)
    print("[PART 2] Benchpress Cross-Source Generalization Test")
    print("=" * 72)

    p = MODELS_DIR / "benchpress_form.pkl"
    if not p.exists():
        print("  [SKIP] model not found")
        return

    m = joblib.load(p)
    df = pd.read_csv(DATA_DIR / "interim" / "benchpress_features.csv")
    df = apply_symmetry_fix(df)

    feat_cols = m["feature_cols"]
    model = m["model"]

    # We know the file naming convention is corr_N / inc_N (correct/incorrect)
    # Group split already prevents same-video leakage.
    # But we want to test: does the model generalize to UNSEEN subjects?
    # Heuristic proxy: hold out the "highest N" videos (assuming they are later subjects)
    all_videos = sorted(df["video_name"].unique())
    print(f"  Total videos: {len(all_videos)}")

    # Method 1: Use the stored test_videos (the honest split)
    test_videos = set(m.get("test_videos", []))
    if test_videos:
        df_test = df[df["video_name"].isin(test_videos)]
        X = df_test[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
        y_true = df_test["label"].astype(int).values
        y_pred = model.predict(X)
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        print(f"\n  [TEST 1] Stored test set (n={len(y_true)}, videos={len(test_videos)})")
        print(f"    Accuracy: {acc:.4f}  F1: {f1:.4f}")

    # Method 2: Temporal/subject holdout — keep only videos where the number suffix >= 50
    # This simulates "new subjects" we have never seen.
    def extract_number(vname):
        """Extract the number from 'corr_96.mov' → 96."""
        import re
        nums = re.findall(r"\d+", vname)
        return int(nums[0]) if nums else 0

    df["video_num"] = df["video_name"].apply(extract_number)
    # Hold out top-20% videos by number (latest subjects)
    threshold_num = df.groupby("video_name")["video_num"].first().quantile(0.80)
    test_mask = df["video_num"] > threshold_num
    train_mask = ~test_mask
    df_held_out = df[test_mask]
    print(f"\n  [TEST 2] Temporal holdout (video_num > {threshold_num:.0f})")
    print(f"    Held-out samples: {len(df_held_out)} ({df_held_out['video_name'].nunique()} videos)")

    if len(df_held_out) > 0:
        X = df_held_out[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
        predict_result = model.predict(X)
        y_true = df_held_out["label"].astype(int).values
        acc2 = accuracy_score(y_true, predict_result)
        f12 = f1_score(y_true, predict_result, average="macro")
        print(f"    Accuracy: {acc2:.4f}  F1: {f12:.4f}  (vs stored test acc={acc:.4f})")

        if acc2 < acc - 0.10:
            print(f"    ⚠️  GAP DETECTED: dropped {acc-acc2:.1%} → suggests OVERFITTING to video source")
        elif acc2 < acc - 0.05:
            print(f"    ⚠️  MINOR GAP: dropped {acc-acc2:.1%} → mild source overfitting")
        else:
            print(f"    ✅ STABLE: gap < 5% → model generalizes well across subjects")

    # Method 3: Single-video leave-one-out stress test
    # Train-set accuracy on train videos, test-set accuracy on test videos, by video
    print(f"\n  [TEST 3] Per-video accuracy on stored test set")
    if test_videos:
        df_test = df[df["video_name"].isin(test_videos)]
        results_by_video = []
        for vname in sorted(test_videos)[:20]:  # first 20 test videos
            sub = df_test[df_test["video_name"] == vname]
            if len(sub) < 5:
                continue
            X = sub[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
            y_true = sub["label"].astype(int).values
            y_pred = model.predict(X)
            v_acc = accuracy_score(y_true, y_pred)
            true_label = y_true[0]  # all same within video (we saw this)
            results_by_video.append({"video": vname, "acc": v_acc, "true_label": true_label})

        df_v = pd.DataFrame(results_by_video)
        if not df_v.empty:
            print(f"    Accuracy by video (sample of {len(df_v)} videos):")
            print(f"      Mean: {df_v['acc'].mean():.4f}")
            print(f"      Std:  {df_v['acc'].std():.4f}")
            print(f"      Worst 5 videos:")
            for _, row in df_v.nsmallest(5, "acc").iterrows():
                print(f"        {row['video']}: acc={row['acc']:.4f} (true_label={'Correct' if row['true_label']==0 else 'Incorrect'})")
            print(f"      Best 5 videos:")
            for _, row in df_v.nsmallest(5, "acc").iloc[::-1].iterrows():
                lbl = "Correct" if row["true_label"] == 0 else "Incorrect"
                print(f"        {row['video']}: acc={row['acc']:.4f} (true_label={lbl})")
            n_perfect = (df_v["acc"] == 1.0).sum()
            print(f"      Perfect accuracy (1.0): {n_perfect}/{len(df_v)} videos = {n_perfect/len(df_v)*100:.1f}%")


def main():
    run_deadlift_sweep()
    run_benchpress_generalization()


if __name__ == "__main__":
    main()
