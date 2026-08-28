# FitVision — Ablation Study Results

> สรุปผลจาก Ablation Study

---

## 1. Deadlift

### Feature Ablation
**Baseline**: acc=0.8467, f1=0.8272

- ลบ `left_elbow_angle`: \u0394acc=+0.0039, \u0394f1=+0.0038 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `right_elbow_angle`: \u0394acc=+0.0064, \u0394f1=+0.0066 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `left_shoulder_angle`: \u0394acc=-0.0034, \u0394f1=-0.0035 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `right_shoulder_angle`: \u0394acc=-0.0020, \u0394f1=-0.0021 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `left_hip_angle`: \u0394acc=-0.0009, \u0394f1=-0.0009 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `right_hip_angle`: \u0394acc=-0.0063, \u0394f1=-0.0071 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `left_knee_angle`: \u0394acc=-0.0062, \u0394f1=-0.0060 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `right_knee_angle`: \u0394acc=-0.0093, \u0394f1=-0.0107 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `shoulder_width`: \u0394acc=+0.0018, \u0394f1=-0.0008 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `hip_width`: \u0394acc=+0.0029, \u0394f1=+0.0028 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `torso_length`: \u0394acc=-0.0357, \u0394f1=-0.0393 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `elbow_symmetry`: \u0394acc=+0.0010, \u0394f1=+0.0008 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `knee_symmetry`: \u0394acc=+0.0014, \u0394f1=+0.0011 (⚪ ไม่สำคัญ / แย่ลง)

### Model Component Ablation
- ** XGBoost_only**: acc=0.8258, f1=0.8066
- ** RandomForest_only**: acc=0.864, f1=0.8434
- ** Ensemble_VotingClassifier**: acc=0.8467, f1=0.8272

### Data Processing Ablation

####  Exp 1: GroupShuffleSplit vs Random Split
-  Group: acc=0.8467, Random: acc=0.9133
-  Leakage inflation: +0.0666

####  Exp 2: SMOTE vs No SMOTE
-  SMOTE: acc=0.8467, No SMOTE: acc=0.8622

####  Exp 3: class_weight='balanced' vs None
-  Balanced: f1=0.8272, None: f1=0.8272

## 2. Benchpress

### Feature Ablation
**Baseline**: acc=0.9939, f1=0.9934

- ลบ `left_elbow_angle`: \u0394acc=-0.0004, \u0394f1=-0.0005 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `right_elbow_angle`: \u0394acc=+0.0004, \u0394f1=+0.0003 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `left_shoulder_angle`: \u0394acc=-0.0003, \u0394f1=-0.0004 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `right_shoulder_angle`: \u0394acc=-0.0004, \u0394f1=-0.0005 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `left_hip_angle`: \u0394acc=-0.0002, \u0394f1=-0.0003 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `right_hip_angle`: \u0394acc=-0.0001, \u0394f1=-0.0002 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `left_knee_angle`: \u0394acc=-0.0008, \u0394f1=-0.0009 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `right_knee_angle`: \u0394acc=-0.0005, \u0394f1=-0.0006 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `shoulder_width`: \u0394acc=-0.0017, \u0394f1=-0.0019 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `hip_width`: \u0394acc=-0.0079, \u0394f1=-0.0087 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `torso_length`: \u0394acc=-0.0011, \u0394f1=-0.0013 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `elbow_symmetry`: \u0394acc=-0.0021, \u0394f1=-0.0024 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `knee_symmetry`: \u0394acc=+0.0000, \u0394f1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)

### Model Component Ablation
- ** XGBoost_only**: acc=0.994, f1=0.9935
- ** RandomForest_only**: acc=0.991, f1=0.9902
- ** Ensemble_VotingClassifier**: acc=0.9939, f1=0.9934

### Data Processing Ablation

####  Exp 1: GroupShuffleSplit vs Random Split
-  Group: acc=0.9939, Random: acc=0.9996
-  Leakage inflation: +0.0057

####  Exp 2: SMOTE vs No SMOTE
-  SMOTE: acc=0.9939, No SMOTE: acc=0.9947

####  Exp 3: class_weight='balanced' vs None
-  Balanced: f1=0.9934, None: f1=0.9934

## 3. Squat

### Feature Ablation
**Baseline**: acc=0.9973, f1=0.997

- ลบ `left_knee_angle`: \u0394acc=+0.0000, \u0394f1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `right_knee_angle`: \u0394acc=-0.0013, \u0394f1=-0.0015 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `left_hip_angle`: \u0394acc=+0.0000, \u0394f1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `right_hip_angle`: \u0394acc=-0.0013, \u0394f1=-0.0015 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `left_ankle_angle`: \u0394acc=-0.0013, \u0394f1=-0.0015 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `right_ankle_angle`: \u0394acc=+0.0000, \u0394f1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `spine_angle`: \u0394acc=+0.0000, \u0394f1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `torso_lean`: \u0394acc=+0.0000, \u0394f1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `left_knee_lateral`: \u0394acc=+0.0000, \u0394f1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `right_knee_lateral`: \u0394acc=+0.0014, \u0394f1=+0.0015 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `symmetry_score`: \u0394acc=-0.0013, \u0394f1=-0.0015 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `hip_depth`: \u0394acc=-0.0026, \u0394f1=-0.0030 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `avg_knee_angle`: \u0394acc=+0.0000, \u0394f1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `avg_hip_angle`: \u0394acc=-0.0013, \u0394f1=-0.0015 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `knee_hip_ratio`: \u0394acc=-0.0053, \u0394f1=-0.0060 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `knee_depth_ratio`: \u0394acc=+0.0014, \u0394f1=+0.0015 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `ankle_asymmetry`: \u0394acc=+0.0000, \u0394f1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `hip_asymmetry`: \u0394acc=-0.0013, \u0394f1=-0.0015 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `total_lateral`: \u0394acc=+0.0014, \u0394f1=+0.0015 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `lean_consistency`: \u0394acc=+0.0000, \u0394f1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)

### Squat Feature Group (Base 12 vs Full 20)

### Model Component Ablation
- ** XGBoost_only**: acc=0.9973, f1=0.997
- ** RandomForest_only**: acc=0.9973, f1=0.997
- ** Ensemble_VotingClassifier**: acc=0.9973, f1=0.997

### Feature Ablation
**Baseline**: acc=0.9973, f1=0.9973

- ลบ `left_knee_angle`: \u0394acc=+0.0000, \u0394f1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `right_knee_angle`: \u0394acc=+0.0000, \u0394f1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `left_hip_angle`: \u0394acc=+0.0000, \u0394f1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `right_hip_angle`: \u0394acc=+0.0000, \u0394f1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `left_ankle_angle`: \u0394acc=-0.0039, \u0394f1=-0.0039 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `right_ankle_angle`: \u0394acc=+0.0014, \u0394f1=+0.0014 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `spine_angle`: \u0394acc=+0.0000, \u0394f1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `torso_lean`: \u0394acc=+0.0000, \u0394f1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `left_knee_lateral`: \u0394acc=+0.0000, \u0394f1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `right_knee_lateral`: \u0394acc=+0.0000, \u0394f1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `symmetry_score`: \u0394acc=+0.0000, \u0394f1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `hip_depth`: \u0394acc=+0.0000, \u0394f1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `avg_knee_angle`: \u0394acc=+0.0000, \u0394f1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `avg_hip_angle`: \u0394acc=+0.0000, \u0394f1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `knee_hip_ratio`: \u0394acc=-0.0039, \u0394f1=-0.0041 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `knee_depth_ratio`: \u0394acc=+0.0000, \u0394f1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `ankle_asymmetry`: \u0394acc=+0.0000, \u0394f1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `hip_asymmetry`: \u0394acc=+0.0000, \u0394f1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `total_lateral`: \u0394acc=+0.0000, \u0394f1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `lean_consistency`: \u0394acc=+0.0000, \u0394f1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)

### Model Component Ablation
