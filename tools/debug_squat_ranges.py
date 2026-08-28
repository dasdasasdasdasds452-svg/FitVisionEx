"""Quick script to analyze squat training data feature ranges"""
import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

df = pd.read_csv("data/raw/kaggle_squat/squat_features_augmented.csv")
print(f"Shape: {df.shape}")
print(f"\nLabel distribution:")
print(df['label'].value_counts().sort_index())

cols = ['left_knee_angle','right_knee_angle','left_hip_angle','right_hip_angle',
        'left_ankle_angle','right_ankle_angle','spine_angle','torso_lean',
        'left_knee_lateral','right_knee_lateral','symmetry_score','hip_depth']

print(f"\n{'Feature':<25s} {'min':>8s} {'mean':>8s} {'max':>8s} {'std':>8s}")
print("-" * 60)
for c in cols:
    mn = df[c].min()
    me = df[c].mean()
    mx = df[c].max()
    st = df[c].std()
    print(f"{c:<25s} {mn:>8.2f} {me:>8.2f} {mx:>8.2f} {st:>8.2f}")

# Show ranges for Correct (label=0) vs Incorrect (label!=0)
print(f"\n\n=== Correct (label=0) feature means ===")
correct = df[df['label'] == 0]
incorrect = df[df['label'] != 0]
print(f"Correct samples: {len(correct)}, Incorrect samples: {len(incorrect)}")
print(f"\n{'Feature':<25s} {'Correct':>10s} {'Incorrect':>10s} {'Diff':>10s}")
print("-" * 60)
for c in cols:
    cm = correct[c].mean()
    im = incorrect[c].mean()
    print(f"{c:<25s} {cm:>10.2f} {im:>10.2f} {cm-im:>10.2f}")
