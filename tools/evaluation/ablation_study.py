"""
FitVision Ablation Study
=========================
ทดสอบอิทธิพลของแต่ละ component ต่อ model performance
3 มิติ: (A) Feature Ablation, (B) Model Component Ablation, (C) Data Processing Ablation
4 โมเดล: squat_binary, squat_3class, deadlift, benchpress

Usage:
    python tools/evaluation/ablation_study.py
    
Output:
    data/evaluation/ablation/
    ├── ablation_results.json          ← ผลลัพธ์ทั้งหมด
    ├── feature_ablation_*.png         ← กราฟ feature importance
    ├── model_ablation_*.png           ← กราฟเปรียบเทียบ model components
    ├── data_processing_ablation_*.png ← กราฟเปรียบเทียบ data processing
    └── ablation_summary.md            ← สรุปเป็น markdown
"""

import json
import logging
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "evaluation" / "ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared Feature Columns ─────────────────────────────────────
FEATURES_13 = [
    'left_elbow_angle', 'right_elbow_angle',
    'left_shoulder_angle', 'right_shoulder_angle',
    'left_hip_angle', 'right_hip_angle',
    'left_knee_angle', 'right_knee_angle',
    'shoulder_width', 'hip_width', 'torso_length',
    'elbow_symmetry', 'knee_symmetry',
]

SQUAT_BASE_12 = [
    'left_knee_angle', 'right_knee_angle',
    'left_hip_angle', 'right_hip_angle',
    'left_ankle_angle', 'right_ankle_angle',
    'spine_angle', 'torso_lean',
    'left_knee_lateral', 'right_knee_lateral',
    'symmetry_score', 'hip_depth',
]

SQUAT_ENGINEERED_8 = [
    'avg_knee_angle', 'avg_hip_angle', 'knee_hip_ratio',
    'knee_depth_ratio', 'ankle_asymmetry', 'hip_asymmetry',
    'total_lateral', 'lean_consistency',
]

SQUAT_FEATURES_20 = SQUAT_BASE_12 + SQUAT_ENGINEERED_8

# ── Plot Style ─────────────────────────────────────────────────
plt.rcParams.update({
    'figure.figsize': (12, 6),
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})
COLORS = {
    'baseline': '#2196F3',
    'ablated': '#F44336',
    'positive': '#4CAF50',
    'neutral': '#9E9E9E',
    'xgb': '#FF9800',
    'rf': '#4CAF50',
    'ensemble': '#2196F3',
}


# ═══════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════

def load_deadlift_data() -> tuple:
    """Load deadlift data from training_dataset.csv."""
    csv_path = PROJECT_ROOT / "data" / "processed" / "training_dataset.csv"
    df = pd.read_csv(csv_path, low_memory=False)
    df = df[df['exercise'] == 'deadlift'].copy()

    # Fix symmetry bug
    df['elbow_symmetry'] = (df['left_elbow_angle'] - df['right_elbow_angle']).abs()
    df['knee_symmetry'] = (df['left_knee_angle'] - df['right_knee_angle']).abs()

    X = df[FEATURES_13].apply(pd.to_numeric, errors='coerce').fillna(0).values
    y = df['form_correct'].astype(int).values
    groups = df['video_name'].values
    return X, y, groups, FEATURES_13, "deadlift"


def load_benchpress_data() -> tuple:
    """Load benchpress data from benchpress_features.csv."""
    csv_path = PROJECT_ROOT / "data" / "interim" / "benchpress_features.csv"
    df = pd.read_csv(csv_path)

    # Fix symmetry bug
    df['elbow_symmetry'] = (df['left_elbow_angle'] - df['right_elbow_angle']).abs()
    df['knee_symmetry'] = (df['left_knee_angle'] - df['right_knee_angle']).abs()

    X = df[FEATURES_13].apply(pd.to_numeric, errors='coerce').fillna(0).values
    y = df['label'].astype(int).values
    groups = df['video_name'].values
    return X, y, groups, FEATURES_13, "benchpress"


def add_squat_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 8 engineered features to squat dataframe."""
    df = df.copy()
    df['avg_knee_angle'] = (df['left_knee_angle'] + df['right_knee_angle']) / 2
    df['avg_hip_angle'] = (df['left_hip_angle'] + df['right_hip_angle']) / 2
    df['knee_hip_ratio'] = df['avg_knee_angle'] / (df['avg_hip_angle'] + 1e-8)
    df['knee_depth_ratio'] = df['avg_knee_angle'] / 90.0
    df['ankle_asymmetry'] = (df['left_ankle_angle'] - df['right_ankle_angle']).abs()
    df['hip_asymmetry'] = (df['left_hip_angle'] - df['right_hip_angle']).abs()
    df['total_lateral'] = df['left_knee_lateral'].abs() + df['right_knee_lateral'].abs()
    df['lean_consistency'] = (df['spine_angle'] - df['torso_lean']).abs()
    return df


def load_squat_data() -> tuple:
    """Load squat data from pre-extracted CSV."""
    csv_path = PROJECT_ROOT / "data" / "processed" / "squat_real_labels.csv"
    df = pd.read_csv(csv_path)
    df = add_squat_engineered_features(df)

    X = df[SQUAT_FEATURES_20].apply(pd.to_numeric, errors='coerce').fillna(0).values
    y_bin = df['binary_label'].astype(int).values    # 0=Good, 1=Bad
    
    # XGBoost requires contiguous labels starting from 0 (e.g. 0, 1, 2 instead of 0, 2, 4)
    y_multi_raw = df['error_code'].astype(int).values
    label_map = {0: 0, 2: 1, 4: 2}
    y_multi = np.array([label_map.get(lbl, 0) for lbl in y_multi_raw])

    split = df['split'].values if 'split' in df.columns else None
    return X, y_bin, y_multi, split, df, SQUAT_FEATURES_20, "squat"


# ═══════════════════════════════════════════════════════════════
# Training Helpers
# ═══════════════════════════════════════════════════════════════

def make_xgb(binary: bool = True, **extra) -> xgb.XGBClassifier:
    """Create XGBClassifier with standard params."""
    params = dict(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss' if binary else 'mlogloss',
        random_state=42, n_jobs=-1,
    )
    params.update(extra)
    return xgb.XGBClassifier(**params)


def make_rf(**extra) -> RandomForestClassifier:
    """Create RandomForestClassifier with standard params."""
    params = dict(
        n_estimators=150, max_depth=15, min_samples_leaf=3,
        class_weight='balanced', random_state=42, n_jobs=-1,
    )
    params.update(extra)
    return RandomForestClassifier(**params)


def make_ensemble(binary: bool = True) -> VotingClassifier:
    """Create standard VotingClassifier (XGB + RF)."""
    return VotingClassifier(
        estimators=[('xgb', make_xgb(binary)), ('rf', make_rf())],
        voting='soft',
    )


def train_and_eval(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    model, use_smote: bool = True,
) -> dict:
    """Train model and return metrics dict."""
    if use_smote and len(np.unique(y_train)) > 1:
        try:
            sm = SMOTE(random_state=42)
            X_fit, y_fit = sm.fit_resample(X_train, y_train)
        except ValueError:
            X_fit, y_fit = X_train, y_train
    else:
        X_fit, y_fit = X_train, y_train

    model.fit(X_fit, y_fit)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    result = {'accuracy': round(acc, 4), 'f1_macro': round(f1, 4)}

    # ROC-AUC for binary
    if len(np.unique(y_test)) == 2:
        try:
            proba = model.predict_proba(X_test)[:, 1]
            result['roc_auc'] = round(roc_auc_score(y_test, proba), 4)
        except Exception:
            pass

    return result


def split_data(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    use_group_split: bool = True,
) -> tuple:
    """Split data with GroupShuffleSplit or random."""
    if use_group_split and groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(X)), test_size=0.2, random_state=42, stratify=y,
        )
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


# ═══════════════════════════════════════════════════════════════
# Ablation Experiments
# ═══════════════════════════════════════════════════════════════

def run_feature_ablation(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    feature_names: list, model_name: str, binary: bool = True,
) -> dict:
    """Leave-one-out feature ablation."""
    log.info(f"[{model_name}] Feature Ablation — {len(feature_names)} features")
    X_train, X_test, y_train, y_test = split_data(X, y, groups)

    # Baseline: all features
    baseline = train_and_eval(X_train, y_train, X_test, y_test, make_ensemble(binary))
    log.info(f"  Baseline: acc={baseline['accuracy']}, f1={baseline['f1_macro']}")

    results = {'baseline': baseline, 'removed_features': {}}

    for i, feat in enumerate(feature_names):
        cols = [j for j in range(len(feature_names)) if j != i]
        m = train_and_eval(
            X_train[:, cols], y_train, X_test[:, cols], y_test,
            make_ensemble(binary),
        )
        delta_acc = round(m['accuracy'] - baseline['accuracy'], 4)
        delta_f1 = round(m['f1_macro'] - baseline['f1_macro'], 4)
        m['delta_accuracy'] = delta_acc
        m['delta_f1_macro'] = delta_f1
        results['removed_features'][feat] = m
        log.info(f"  Remove '{feat}': Δacc={delta_acc:+.4f}, Δf1={delta_f1:+.4f}")

    return results


def run_feature_group_ablation_squat(
    X: np.ndarray, y: np.ndarray,
    X_train: np.ndarray, X_test: np.ndarray,
    y_train: np.ndarray, y_test: np.ndarray,
    binary: bool = True,
) -> dict:
    """Test base 12 features vs full 20 features for squat."""
    log.info(f"  Squat Feature Group: base-12 vs full-20")

    # Full 20 features (baseline)
    full = train_and_eval(X_train, y_train, X_test, y_test, make_ensemble(binary))

    # Base 12 features only (remove last 8 engineered columns)
    base_train, base_test = X_train[:, :12], X_test[:, :12]
    base = train_and_eval(base_train, y_train, base_test, y_test, make_ensemble(binary))

    return {
        'full_20_features': full,
        'base_12_features': base,
        'engineered_delta_accuracy': round(full['accuracy'] - base['accuracy'], 4),
        'engineered_delta_f1': round(full['f1_macro'] - base['f1_macro'], 4),
    }


def run_model_ablation(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    model_name: str, binary: bool = True,
) -> dict:
    """Compare XGB only vs RF only vs Ensemble."""
    log.info(f"[{model_name}] Model Component Ablation")
    X_train, X_test, y_train, y_test = split_data(X, y, groups)

    results = {}
    configs = {
        'XGBoost_only': make_xgb(binary),
        'RandomForest_only': make_rf(),
        'Ensemble_VotingClassifier': make_ensemble(binary),
    }

    for name, model in configs.items():
        m = train_and_eval(X_train, y_train, X_test, y_test, model)
        results[name] = m
        log.info(f"  {name}: acc={m['accuracy']}, f1={m['f1_macro']}")

    return results


def run_data_processing_ablation(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    model_name: str, binary: bool = True,
) -> dict:
    """Test SMOTE/no-SMOTE, balanced/unbalanced, GroupSplit/RandomSplit."""
    log.info(f"[{model_name}] Data Processing Ablation")
    results = {}

    # ── Experiment 1: GroupShuffleSplit vs Random Split ──
    log.info(f"  Exp 1: GroupShuffleSplit vs Random Split")
    X_tr_g, X_te_g, y_tr_g, y_te_g = split_data(X, y, groups, use_group_split=True)
    X_tr_r, X_te_r, y_tr_r, y_te_r = split_data(X, y, groups, use_group_split=False)

    group_res = train_and_eval(X_tr_g, y_tr_g, X_te_g, y_te_g, make_ensemble(binary))
    random_res = train_and_eval(X_tr_r, y_tr_r, X_te_r, y_te_r, make_ensemble(binary))
    results['split_method'] = {
        'GroupShuffleSplit': group_res,
        'Random_train_test_split': random_res,
        'leakage_inflation': round(random_res['accuracy'] - group_res['accuracy'], 4),
    }
    log.info(f"    Group: acc={group_res['accuracy']}, Random: acc={random_res['accuracy']}")
    log.info(f"    Leakage inflation: {results['split_method']['leakage_inflation']:+.4f}")

    # ── Experiment 2: With SMOTE vs Without SMOTE ──
    log.info(f"  Exp 2: SMOTE vs No SMOTE")
    X_tr, X_te, y_tr, y_te = split_data(X, y, groups, use_group_split=True)
    smote_res = train_and_eval(X_tr, y_tr, X_te, y_te, make_ensemble(binary), use_smote=True)
    no_smote_res = train_and_eval(X_tr, y_tr, X_te, y_te, make_ensemble(binary), use_smote=False)
    results['smote'] = {
        'with_SMOTE': smote_res,
        'without_SMOTE': no_smote_res,
        'smote_delta_accuracy': round(smote_res['accuracy'] - no_smote_res['accuracy'], 4),
        'smote_delta_f1': round(smote_res['f1_macro'] - no_smote_res['f1_macro'], 4),
    }
    log.info(f"    SMOTE: acc={smote_res['accuracy']}, No SMOTE: acc={no_smote_res['accuracy']}")

    # ── Experiment 3: class_weight='balanced' vs None ──
    log.info(f"  Exp 3: class_weight='balanced' vs None")
    balanced_ens = VotingClassifier(
        estimators=[
            ('xgb', make_xgb(binary)),
            ('rf', make_rf(class_weight='balanced')),
        ], voting='soft',
    )
    unbalanced_ens = VotingClassifier(
        estimators=[
            ('xgb', make_xgb(binary)),
            ('rf', make_rf(class_weight=None)),
        ], voting='soft',
    )
    balanced_res = train_and_eval(X_tr, y_tr, X_te, y_te, balanced_ens)
    unbalanced_res = train_and_eval(X_tr, y_tr, X_te, y_te, unbalanced_ens)
    results['class_weight'] = {
        'balanced': balanced_res,
        'none': unbalanced_res,
        'balanced_delta_f1': round(balanced_res['f1_macro'] - unbalanced_res['f1_macro'], 4),
    }
    log.info(f"    Balanced: f1={balanced_res['f1_macro']}, None: f1={unbalanced_res['f1_macro']}")

    return results


# ═══════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════

def plot_feature_ablation(results: dict, model_name: str) -> None:
    """Bar chart showing accuracy drop when each feature is removed."""
    features = results['removed_features']
    names = list(features.keys())
    deltas = [features[f]['delta_f1_macro'] for f in names]

    # Sort by impact (most negative first = most important)
    sorted_idx = np.argsort(deltas)
    names = [names[i] for i in sorted_idx]
    deltas = [deltas[i] for i in sorted_idx]

    colors = [COLORS['ablated'] if d < -0.005 else
              COLORS['positive'] if d > 0.005 else
              COLORS['neutral'] for d in deltas]

    fig, ax = plt.subplots(figsize=(12, max(6, len(names) * 0.4)))
    bars = ax.barh(names, deltas, color=colors, edgecolor='white', linewidth=0.5)
    ax.axvline(x=0, color='black', linewidth=0.8)

    for bar, delta in zip(bars, deltas):
        offset = -0.003 if delta < 0 else 0.001
        ax.text(bar.get_width() + offset, bar.get_y() + bar.get_height() / 2,
                f'{delta:+.4f}', va='center', fontsize=9)

    ax.set_xlabel('ΔF1-macro (เมื่อลบ feature ออก)')
    ax.set_title(f'Feature Ablation — {model_name}\n(ค่าลบ = feature สำคัญ, ค่าบวก = feature ไม่จำเป็น)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'feature_ablation_{model_name}.png', dpi=150)
    plt.close()
    log.info(f"  Saved feature_ablation_{model_name}.png")


def plot_model_ablation(results: dict, model_name: str) -> None:
    """Grouped bar chart for model components."""
    names = list(results.keys())
    acc = [results[n]['accuracy'] for n in names]
    f1 = [results[n]['f1_macro'] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, acc, width, label='Accuracy', color=COLORS['baseline'])
    bars2 = ax.bar(x + width / 2, f1, width, label='F1-macro', color=COLORS['positive'])

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=9)

    ax.set_ylabel('Score')
    ax.set_title(f'Model Component Ablation — {model_name}')
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace('_', '\n') for n in names])
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'model_ablation_{model_name}.png', dpi=150)
    plt.close()
    log.info(f"  Saved model_ablation_{model_name}.png")


def plot_data_processing_ablation(results: dict, model_name: str) -> None:
    """Multi-panel bar chart for data processing ablation."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Split method
    ax = axes[0]
    split = results['split_method']
    names = ['GroupShuffle\nSplit', 'Random\nSplit']
    vals = [split['GroupShuffleSplit']['accuracy'], split['Random_train_test_split']['accuracy']]
    colors = [COLORS['positive'], COLORS['ablated']]
    bars = ax.bar(names, vals, color=colors, edgecolor='white')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{bar.get_height():.4f}', ha='center', fontsize=10)
    ax.set_title(f'Data Split Method\nLeakage inflation: {split["leakage_inflation"]:+.4f}')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.1)

    # Panel 2: SMOTE
    ax = axes[1]
    smote = results['smote']
    names = ['With\nSMOTE', 'Without\nSMOTE']
    vals = [smote['with_SMOTE']['f1_macro'], smote['without_SMOTE']['f1_macro']]
    colors = [COLORS['positive'], COLORS['ablated']]
    bars = ax.bar(names, vals, color=colors, edgecolor='white')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{bar.get_height():.4f}', ha='center', fontsize=10)
    ax.set_title(f'SMOTE Effect\nΔF1: {smote["smote_delta_f1"]:+.4f}')
    ax.set_ylabel('F1-macro')
    ax.set_ylim(0, 1.1)

    # Panel 3: class_weight
    ax = axes[2]
    cw = results['class_weight']
    names = ['class_weight\n=balanced', 'class_weight\n=None']
    vals = [cw['balanced']['f1_macro'], cw['none']['f1_macro']]
    colors = [COLORS['positive'], COLORS['ablated']]
    bars = ax.bar(names, vals, color=colors, edgecolor='white')
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{bar.get_height():.4f}', ha='center', fontsize=10)
    ax.set_title(f'Class Weight Effect\nΔF1: {cw["balanced_delta_f1"]:+.4f}')
    ax.set_ylabel('F1-macro')
    ax.set_ylim(0, 1.1)

    fig.suptitle(f'Data Processing Ablation — {model_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'data_processing_ablation_{model_name}.png', dpi=150)
    plt.close()
    log.info(f"  Saved data_processing_ablation_{model_name}.png")


def plot_squat_feature_group(results: dict) -> None:
    """Bar chart for squat base-12 vs full-20 features."""
    fig, ax = plt.subplots(figsize=(8, 5))
    names = ['Base 12\nFeatures', 'Full 20\nFeatures\n(+8 Engineered)']
    acc = [results['base_12_features']['accuracy'], results['full_20_features']['accuracy']]
    f1 = [results['base_12_features']['f1_macro'], results['full_20_features']['f1_macro']]

    x = np.arange(len(names))
    width = 0.35
    bars1 = ax.bar(x - width / 2, acc, width, label='Accuracy', color=COLORS['baseline'])
    bars2 = ax.bar(x + width / 2, f1, width, label='F1-macro', color=COLORS['positive'])

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{bar.get_height():.4f}', ha='center', fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{bar.get_height():.4f}', ha='center', fontsize=10)

    delta_a = results['engineered_delta_accuracy']
    delta_f = results['engineered_delta_f1']
    ax.set_title(f'Squat Feature Group Ablation\nEngineered Features: Δacc={delta_a:+.4f}, ΔF1={delta_f:+.4f}')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'feature_group_ablation_squat.png', dpi=150)
    plt.close()
    log.info(f"  Saved feature_group_ablation_squat.png")


# ═══════════════════════════════════════════════════════════════
# Summary Report
# ═══════════════════════════════════════════════════════════════

def generate_summary_md(all_results: dict) -> None:
    """Generate markdown summary of ablation study."""
    lines = [
        "# FitVision — Ablation Study Results",
        "",
        f"> วันที่: {time.strftime('%Y-%m-%d')}",
        f"> Script: `tools/evaluation/ablation_study.py`",
        "",
        "---",
        "",
    ]

    for model_name, res in all_results.items():
        lines.append(f"## {model_name}")
        lines.append("")

        # Feature ablation
        if 'feature_ablation' in res:
            fa = res['feature_ablation']
            baseline = fa['baseline']
            lines.append(f"### Feature Ablation (Baseline: acc={baseline['accuracy']}, f1={baseline['f1_macro']})")
            lines.append("")
            lines.append("| Feature Removed | Accuracy | F1-macro | Δacc | ΔF1 | สำคัญ? |")
            lines.append("|---|---:|---:|---:|---:|---|")

            # Sort by impact
            feats = fa['removed_features']
            sorted_feats = sorted(feats.items(), key=lambda x: x[1]['delta_f1_macro'])
            for feat_name, m in sorted_feats:
                flag = "🔴 สำคัญมาก" if m['delta_f1_macro'] < -0.01 else \
                       "🟡 สำคัญ" if m['delta_f1_macro'] < -0.003 else \
                       "⚪ ไม่สำคัญ" if m['delta_f1_macro'] > 0.003 else "🟢 เล็กน้อย"
                lines.append(
                    f"| `{feat_name}` | {m['accuracy']} | {m['f1_macro']} "
                    f"| {m['delta_accuracy']:+.4f} | {m['delta_f1_macro']:+.4f} | {flag} |"
                )
            lines.append("")

        # Feature group (squat only)
        if 'feature_group_ablation' in res:
            fg = res['feature_group_ablation']
            lines.append("### Feature Group Ablation (Base-12 vs Full-20)")
            lines.append("")
            lines.append("| Feature Set | Accuracy | F1-macro |")
            lines.append("|---|---:|---:|")
            lines.append(f"| Base 12 Features | {fg['base_12_features']['accuracy']} | {fg['base_12_features']['f1_macro']} |")
            lines.append(f"| Full 20 Features (+8 Engineered) | {fg['full_20_features']['accuracy']} | {fg['full_20_features']['f1_macro']} |")
            lines.append(f"| **Δ (Engineered contribution)** | **{fg['engineered_delta_accuracy']:+.4f}** | **{fg['engineered_delta_f1']:+.4f}** |")
            lines.append("")

        # Model ablation
        if 'model_ablation' in res:
            ma = res['model_ablation']
            lines.append("### Model Component Ablation")
            lines.append("")
            lines.append("| Model | Accuracy | F1-macro |")
            lines.append("|---|---:|---:|")
            for name, m in ma.items():
                lines.append(f"| {name} | {m['accuracy']} | {m['f1_macro']} |")
            lines.append("")

        # Data processing ablation
        if 'data_processing_ablation' in res:
            dp = res['data_processing_ablation']
            lines.append("### Data Processing Ablation")
            lines.append("")

            # Split
            sp = dp['split_method']
            lines.append("#### Split Method")
            lines.append(f"- GroupShuffleSplit: acc={sp['GroupShuffleSplit']['accuracy']}")
            lines.append(f"- Random split: acc={sp['Random_train_test_split']['accuracy']}")
            lines.append(f"- **Leakage inflation: {sp['leakage_inflation']:+.4f}**")
            lines.append("")

            # SMOTE
            sm = dp['smote']
            lines.append("#### SMOTE")
            lines.append(f"- With SMOTE: f1={sm['with_SMOTE']['f1_macro']}")
            lines.append(f"- Without SMOTE: f1={sm['without_SMOTE']['f1_macro']}")
            lines.append(f"- **SMOTE effect: ΔF1={sm['smote_delta_f1']:+.4f}**")
            lines.append("")

            # class_weight
            cw = dp['class_weight']
            lines.append("#### Class Weight")
            lines.append(f"- balanced: f1={cw['balanced']['f1_macro']}")
            lines.append(f"- none: f1={cw['none']['f1_macro']}")
            lines.append(f"- **Balanced effect: ΔF1={cw['balanced_delta_f1']:+.4f}**")
            lines.append("")

        lines.append("---")
        lines.append("")

    summary_path = OUTPUT_DIR / "ablation_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Saved ablation_summary.md")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    t0 = time.time()
    all_results = {}

    # ── 1. Deadlift ───────────────────────────────────────────
    log.info("=" * 60)
    log.info("DEADLIFT")
    log.info("=" * 60)
    X, y, groups, feat_names, name = load_deadlift_data()
    log.info(f"  Data: {X.shape[0]} samples, {X.shape[1]} features")

    dl_res = {}
    dl_res['feature_ablation'] = run_feature_ablation(X, y, groups, feat_names, name)
    plot_feature_ablation(dl_res['feature_ablation'], name)

    dl_res['model_ablation'] = run_model_ablation(X, y, groups, name)
    plot_model_ablation(dl_res['model_ablation'], name)

    dl_res['data_processing_ablation'] = run_data_processing_ablation(X, y, groups, name)
    plot_data_processing_ablation(dl_res['data_processing_ablation'], name)

    all_results['deadlift'] = dl_res

    # ── 2. Benchpress ─────────────────────────────────────────
    log.info("=" * 60)
    log.info("BENCHPRESS")
    log.info("=" * 60)
    X, y, groups, feat_names, name = load_benchpress_data()
    log.info(f"  Data: {X.shape[0]} samples, {X.shape[1]} features")

    bp_res = {}
    bp_res['feature_ablation'] = run_feature_ablation(X, y, groups, feat_names, name)
    plot_feature_ablation(bp_res['feature_ablation'], name)

    bp_res['model_ablation'] = run_model_ablation(X, y, groups, name)
    plot_model_ablation(bp_res['model_ablation'], name)

    bp_res['data_processing_ablation'] = run_data_processing_ablation(X, y, groups, name)
    plot_data_processing_ablation(bp_res['data_processing_ablation'], name)

    all_results['benchpress'] = bp_res

    # ── 3. Squat ──────────────────────────────────────────────
    log.info("=" * 60)
    log.info("SQUAT")
    log.info("=" * 60)
    X, y_bin, y_multi, split_col, df_squat, feat_names, name = load_squat_data()
    log.info(f"  Data: {X.shape[0]} samples, {X.shape[1]} features")

    # Use pre-split train/test if available, otherwise synthesize groups
    if split_col is not None:
        train_mask = split_col == 'train'
        test_mask = split_col == 'test'
        X_train_sq, X_test_sq = X[train_mask], X[test_mask]
    else:
        # Fallback: 80/20 random split
        train_mask = np.random.RandomState(42).rand(len(X)) < 0.8
        test_mask = ~train_mask
        X_train_sq, X_test_sq = X[train_mask], X[test_mask]

    # ── 3a. Squat Binary ──
    log.info("--- Squat Binary ---")
    y_tr_bin, y_te_bin = y_bin[train_mask], y_bin[test_mask]

    sq_bin_res = {}

    # Feature ablation (use dummy groups for split)
    dummy_groups = np.arange(len(X))  # each sample = its own "group"
    sq_bin_res['feature_ablation'] = run_feature_ablation(
        X, y_bin, dummy_groups, feat_names, "squat_binary", binary=True)
    plot_feature_ablation(sq_bin_res['feature_ablation'], "squat_binary")

    # Feature group ablation (base-12 vs full-20)
    sq_bin_res['feature_group_ablation'] = run_feature_group_ablation_squat(
        X, y_bin, X_train_sq, X_test_sq, y_tr_bin, y_te_bin, binary=True)
    plot_squat_feature_group(sq_bin_res['feature_group_ablation'])

    # Model ablation
    sq_bin_res['model_ablation'] = run_model_ablation(
        X, y_bin, dummy_groups, "squat_binary", binary=True)
    plot_model_ablation(sq_bin_res['model_ablation'], "squat_binary")

    all_results['squat_binary'] = sq_bin_res

    # ── 3b. Squat 3-Class ──
    log.info("--- Squat 3-Class ---")
    y_tr_multi, y_te_multi = y_multi[train_mask], y_multi[test_mask]

    sq_mc_res = {}
    sq_mc_res['feature_ablation'] = run_feature_ablation(
        X, y_multi, dummy_groups, feat_names, "squat_3class", binary=False)
    plot_feature_ablation(sq_mc_res['feature_ablation'], "squat_3class")

    sq_mc_res['model_ablation'] = run_model_ablation(
        X, y_multi, dummy_groups, "squat_3class", binary=False)
    plot_model_ablation(sq_mc_res['model_ablation'], "squat_3class")

    all_results['squat_3class'] = sq_mc_res

    # ── Save all results ──────────────────────────────────────
    json_path = OUTPUT_DIR / "ablation_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log.info(f"Saved ablation_results.json")

    # ── Generate summary ──────────────────────────────────────
    generate_summary_md(all_results)

    elapsed = time.time() - t0
    log.info(f"Done! Total time: {elapsed / 60:.1f} minutes")
    log.info(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
