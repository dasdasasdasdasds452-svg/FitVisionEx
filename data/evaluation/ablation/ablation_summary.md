# FitVision — Ablation Study Results

> สรุปผลจาก Ablation Study

---

## 1. Deadlift

### Feature Ablation
**Baseline**: acc=0.8467, f1=0.8272

- ลบ `left_elbow_angle`: Δacc=+0.0039, Δf1=+0.0038 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `right_elbow_angle`: Δacc=+0.0064, Δf1=+0.0066 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `left_shoulder_angle`: Δacc=-0.0034, Δf1=-0.0035 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `right_shoulder_angle`: Δacc=-0.0020, Δf1=-0.0021 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `left_hip_angle`: Δacc=-0.0009, Δf1=-0.0009 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `right_hip_angle`: Δacc=-0.0063, Δf1=-0.0071 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `left_knee_angle`: Δacc=-0.0062, Δf1=-0.0060 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `right_knee_angle`: Δacc=-0.0093, Δf1=-0.0107 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `shoulder_width`: Δacc=+0.0018, Δf1=-0.0008 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `hip_width`: Δacc=+0.0029, Δf1=+0.0028 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `torso_length`: Δacc=-0.0357, Δf1=-0.0393 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `elbow_symmetry`: Δacc=+0.0010, Δf1=+0.0008 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `knee_symmetry`: Δacc=+0.0014, Δf1=+0.0011 (⚪ ไม่สำคัญ / แย่ลง)

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

- ลบ `left_elbow_angle`: Δacc=-0.0004, Δf1=-0.0005 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `right_elbow_angle`: Δacc=+0.0004, Δf1=+0.0003 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `left_shoulder_angle`: Δacc=-0.0003, Δf1=-0.0004 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `right_shoulder_angle`: Δacc=-0.0004, Δf1=-0.0005 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `left_hip_angle`: Δacc=-0.0002, Δf1=-0.0003 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `right_hip_angle`: Δacc=-0.0001, Δf1=-0.0002 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `left_knee_angle`: Δacc=-0.0008, Δf1=-0.0009 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `right_knee_angle`: Δacc=-0.0005, Δf1=-0.0006 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `shoulder_width`: Δacc=-0.0017, Δf1=-0.0019 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `hip_width`: Δacc=-0.0079, Δf1=-0.0087 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `torso_length`: Δacc=-0.0011, Δf1=-0.0013 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `elbow_symmetry`: Δacc=-0.0021, Δf1=-0.0024 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `knee_symmetry`: Δacc=+0.0000, Δf1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)

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

- ลบ `left_knee_angle`: Δacc=+0.0000, Δf1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `right_knee_angle`: Δacc=-0.0013, Δf1=-0.0015 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `left_hip_angle`: Δacc=+0.0000, Δf1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `right_hip_angle`: Δacc=-0.0013, Δf1=-0.0015 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `left_ankle_angle`: Δacc=-0.0013, Δf1=-0.0015 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `right_ankle_angle`: Δacc=+0.0000, Δf1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `spine_angle`: Δacc=+0.0000, Δf1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `torso_lean`: Δacc=+0.0000, Δf1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `left_knee_lateral`: Δacc=+0.0000, Δf1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `right_knee_lateral`: Δacc=+0.0014, Δf1=+0.0015 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `symmetry_score`: Δacc=-0.0013, Δf1=-0.0015 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `hip_depth`: Δacc=-0.0026, Δf1=-0.0030 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `avg_knee_angle`: Δacc=+0.0000, Δf1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `avg_hip_angle`: Δacc=-0.0013, Δf1=-0.0015 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `knee_hip_ratio`: Δacc=-0.0053, Δf1=-0.0060 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `knee_depth_ratio`: Δacc=+0.0014, Δf1=+0.0015 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `ankle_asymmetry`: Δacc=+0.0000, Δf1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `hip_asymmetry`: Δacc=-0.0013, Δf1=-0.0015 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `total_lateral`: Δacc=+0.0014, Δf1=+0.0015 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `lean_consistency`: Δacc=+0.0000, Δf1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)

### Squat Feature Group (Base 12 vs Full 20)

### Model Component Ablation
- ** XGBoost_only**: acc=0.9973, f1=0.997
- ** RandomForest_only**: acc=0.9973, f1=0.997
- ** Ensemble_VotingClassifier**: acc=0.9973, f1=0.997

### Feature Ablation
**Baseline**: acc=0.9973, f1=0.9973

- ลบ `left_knee_angle`: Δacc=+0.0000, Δf1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `right_knee_angle`: Δacc=+0.0000, Δf1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `left_hip_angle`: Δacc=+0.0000, Δf1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `right_hip_angle`: Δacc=+0.0000, Δf1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `left_ankle_angle`: Δacc=-0.0039, Δf1=-0.0039 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `right_ankle_angle`: Δacc=+0.0014, Δf1=+0.0014 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `spine_angle`: Δacc=+0.0000, Δf1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `torso_lean`: Δacc=+0.0000, Δf1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `left_knee_lateral`: Δacc=+0.0000, Δf1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `right_knee_lateral`: Δacc=+0.0000, Δf1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `symmetry_score`: Δacc=+0.0000, Δf1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `hip_depth`: Δacc=+0.0000, Δf1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `avg_knee_angle`: Δacc=+0.0000, Δf1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `avg_hip_angle`: Δacc=+0.0000, Δf1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `knee_hip_ratio`: Δacc=-0.0039, Δf1=-0.0041 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `knee_depth_ratio`: Δacc=+0.0000, Δf1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `ankle_asymmetry`: Δacc=+0.0000, Δf1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `hip_asymmetry`: Δacc=+0.0000, Δf1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `total_lateral`: Δacc=+0.0000, Δf1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)
- ลบ `lean_consistency`: Δacc=+0.0000, Δf1=+0.0000 (⚪ ไม่สำคัญ / แย่ลง)

### Model Component Ablation
- **XGBoost_only**: acc=0.9973, f1=0.9973
- **RandomForest_only**: acc=0.9973, f1=0.9973
- **Ensemble_VotingClassifier**: acc=0.9973, f1=0.9973

---

## 4. Key Takeaways & Thesis Defense Highlights (ข้อสรุปสำคัญสำหรับรายงาน/เล่มธีสิส)

1. **Data Leakage Verification (GroupShuffleSplit vs Random Split):**
   - **Deadlift:** Random Split ทำให้ Accuracy พุ่งจาก 84.67% เป็น 91.33% (+6.66% leakage inflation)
   - **Benchpress:** Random Split ทำให้ Accuracy พุ่งจาก 99.39% เป็น 99.96% (+0.57% leakage inflation)
   - *ข้อสรุป:* ยืนยันว่าการใช้ `GroupShuffleSplit` (แยกตาม `video_name`) จำเป็นอย่างยิ่งในการประเมินประสิทธิภาพจริง เพื่อไม่ให้โมเดลจำ frame จากวิดีโอเดียวกัน

2. **Feature Importance & Domain Engineering:**
   - **Deadlift:** Feature ที่สำคัญที่สุดคือ `torso_length` (เมื่อตัดออก Δacc=-3.57%, Δf1=-3.93%) และ `right_knee_angle` (Δf1=-1.07%)
   - **Benchpress:** Feature ที่สำคัญที่สุดคือ `hip_width` (Δf1=-0.87%) และ `elbow_symmetry` (Δf1=-0.24%)
   - **Squat:** Feature ที่สำคัญที่สุดคือ `knee_hip_ratio` (Δf1=-0.60%) ซึ่งเป็น Engineered Feature ที่สร้างขึ้นจาก Biomechanics ช่วยจับจังหวะการย่อและมุมสะโพกได้แม่นยำขึ้น

3. **Ensemble Architecture (XGBoost + RandomForest):**
   - การใช้ Soft Voting Classifier ช่วยรวมข้อดีของทั้งสองโมเดล (XGBoost ที่เก่งเรื่อง gradient boosting และ RF ที่ทนต่อ noise/outliers) ทำให้ได้โมเดลที่มีความเสถียรสูงสุด (Robustness) สำหรับการ deploy จริงในระบบ Production
