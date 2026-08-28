# วิธีการทำงานของ 3 โมเดล (Model Workflows)
## FitVision — ระบบวิเคราะห์ท่าออกกำลังกาย

---

## ขั้นตอนหลัก (General Pipeline) — ใช้ร่วมกันทุกโมเดล

```
วิดีโอ ──▶ [1] YOLOv8 ตรวจจับคน ──▶ [2] Crop ROI ──▶ [3] MediaPipe Pose ──▶ [4] สกัด Features ──▶ [5] ML Classification ──▶ ผลลัพธ์
```

| ขั้นตอน | รายละเอียด | เครื่องมือ |
|:---:|---|---|
| **1. ตรวจจับบุคคล** | ใช้ YOLOv8s (Pre-trained) ตรวจจับตำแหน่งคนในเฟรม วาด Bounding Box แล้ว Crop เฉพาะส่วน ROI (Region of Interest) เพื่อตัด Background ออก | `YOLOv8s` |
| **2. ประมาณท่าทาง** | ส่งภาพ ROI เข้า MediaPipe Pose ระบุตำแหน่ง Landmark 33 จุดบนร่างกาย (ไหล่, ข้อศอก, ข้อมือ, สะโพก, เข่า, ข้อเท้า ฯลฯ) ได้พิกัด (x, y) | `MediaPipe`, `OpenCV` |
| **3. สกัด Features** | คำนวณ Features จาก Landmark ด้วยสูตรตรีโกณมิติ (มุมข้อต่อ) และ Euclidean Distance (ระยะทาง) — **ขั้นตอนนี้แตกต่างกันในแต่ละท่า** | `NumPy` |
| **4. จำแนกท่าทาง** | นำ Features ที่ได้ส่งเข้าโมเดล Machine Learning ที่เทรนไว้ จำแนกว่าท่า **ถูกต้อง (Correct)** หรือ **ไม่ถูกต้อง (Incorrect)** | `scikit-learn` |

---

## 1. โมเดล Squat (สควอท)

### 📐 Feature Extraction — สกัด 20 Features

โมเดล Squat ใช้ Feature Extractor เฉพาะท่า (`SquatFeatureExtractor`) ที่ออกแบบมาเพื่อวิเคราะห์กลไกการนั่งย่อโดยเฉพาะ

#### 12 Base Features:
| # | Feature | คำอธิบาย | วิธีคำนวณ |
|---|---------|---------|----------|
| 1 | `left_knee_angle` | มุมเข่าซ้าย | มุมที่จุดเข่า ระหว่าง สะโพก–เข่า–ข้อเท้า |
| 2 | `right_knee_angle` | มุมเข่าขวา | มุมที่จุดเข่า ระหว่าง สะโพก–เข่า–ข้อเท้า |
| 3 | `left_hip_angle` | มุมสะโพกซ้าย | มุมที่จุดสะโพก ระหว่าง ไหล่–สะโพก–เข่า |
| 4 | `right_hip_angle` | มุมสะโพกขวา | มุมที่จุดสะโพก ระหว่าง ไหล่–สะโพก–เข่า |
| 5 | `left_ankle_angle` | มุมข้อเท้าซ้าย | มุมที่จุดข้อเท้า ระหว่าง เข่า–ข้อเท้า–ปลายเท้า |
| 6 | `right_ankle_angle` | มุมข้อเท้าขวา | มุมที่จุดข้อเท้า ระหว่าง เข่า–ข้อเท้า–ปลายเท้า |
| 7 | `spine_angle` | มุมกระดูกสันหลัง | มุมของ mid_hip → mid_shoulder เทียบกับแนวตั้ง |
| 8 | `torso_lean` | องศาลำตัวเอียง | เท่ากับ spine_angle (วัดการเอียงไปข้างหน้า) |
| 9 | `left_knee_lateral` | การเบี่ยงเข่าซ้าย | ตำแหน่ง x ของเข่า − ตำแหน่ง x ของข้อเท้า |
| 10 | `right_knee_lateral` | การเบี่ยงเข่าขวา | ตำแหน่ง x ของข้อเท้า − ตำแหน่ง x ของเข่า |
| 11 | `symmetry_score` | คะแนนความสมมาตร | ผลรวมของ |มุมเข่าซ้าย−ขวา| + |มุมสะโพกซ้าย−ขวา| |
| 12 | `hip_depth` | ความลึกของสะโพก | ค่า y-coordinate ของ mid_hip (ยิ่งลึกยิ่ง Squat ต่ำ) |

#### 8 Engineered Features (คำนวณจาก Base Features):
| # | Feature | คำอธิบาย | สูตร |
|---|---------|---------|------|
| 13 | `avg_knee_angle` | มุมเข่าเฉลี่ย | (left_knee + right_knee) / 2 |
| 14 | `avg_hip_angle` | มุมสะโพกเฉลี่ย | (left_hip + right_hip) / 2 |
| 15 | `knee_hip_ratio` | อัตราส่วนเข่า/สะโพก | avg_knee / avg_hip |
| 16 | `knee_depth_ratio` | อัตราส่วนความลึก | avg_knee / 90° |
| 17 | `ankle_asymmetry` | ความไม่สมมาตรข้อเท้า | |left_ankle − right_ankle| |
| 18 | `hip_asymmetry` | ความไม่สมมาตรสะโพก | |left_hip − right_hip| |
| 19 | `total_lateral` | ผลรวมการเบี่ยงเข่า | |left_lateral| + |right_lateral| |
| 20 | `lean_consistency` | ความคงเส้นคงวาการเอียง | |spine_angle − torso_lean| |

### 🤖 Classification — 2-Stage Pipeline (จำแนก 2 ขั้นตอน)

```
Features (20 ค่า)
      │
      ├──▶ Stage 1: Binary Model (squat_form.pkl)
      │         ├── ✅ Correct (ท่าถูกต้อง)
      │         └── ❌ Incorrect (ท่าผิด) ──▶ ดูผล Stage 2
      │
      └──▶ Stage 2: Multi-class Model (squat_form_detailed.pkl)
                ├── 0: Correct
                ├── 1: Shallow Squat (ย่อตัวไม่พอ)
                ├── 2: Forward Lean (เอียงตัวไปข้างหน้า)
                ├── 3: Knees Caving In (เข่าบุบเข้าใน)
                ├── 4: Heels Off Ground (ส้นเท้าลอย)
                └── 5: Asymmetric (ไม่สมมาตร)
```

- **Stage 1** — 5-Model VotingClassifier (Soft): XGBoost + LightGBM + CatBoost + RandomForest + GradientBoosting
- **Stage 2** — 4-Model VotingClassifier (Soft): XGBoost + LightGBM + CatBoost + RandomForest
- ใช้ **SMOTE** เพื่อ Balance ข้อมูลก่อนเทรน

---

## 2. โมเดล Deadlift (เดดลิฟท์)

### 📐 Feature Extraction — สกัด 13 Features

โมเดล Deadlift ใช้ `FeatureExtractor` ร่วมกับ Bench Press โดยเน้นมุมข้อต่อทั้งตัวและความสมมาตร

| # | Feature | คำอธิบาย | วิธีคำนวณ |
|---|---------|---------|----------|
| **8 มุมข้อต่อ (Joint Angles)** |||
| 1 | `left_elbow_angle` | มุมข้อศอกซ้าย | ไหล่–ข้อศอก–ข้อมือ |
| 2 | `right_elbow_angle` | มุมข้อศอกขวา | ไหล่–ข้อศอก–ข้อมือ |
| 3 | `left_shoulder_angle` | มุมไหล่ซ้าย | ข้อศอก–ไหล่–สะโพก |
| 4 | `right_shoulder_angle` | มุมไหล่ขวา | ข้อศอก–ไหล่–สะโพก |
| 5 | `left_hip_angle` | มุมสะโพกซ้าย | ไหล่–สะโพก–เข่า |
| 6 | `right_hip_angle` | มุมสะโพกขวา | ไหล่–สะโพก–เข่า |
| 7 | `left_knee_angle` | มุมเข่าซ้าย | สะโพก–เข่า–ข้อเท้า |
| 8 | `right_knee_angle` | มุมเข่าขวา | สะโพก–เข่า–ข้อเท้า |
| **3 ระยะทาง (Distances)** |||
| 9 | `shoulder_width` | ความกว้างไหล่ | Euclidean(ไหล่ซ้าย, ไหล่ขวา) |
| 10 | `hip_width` | ความกว้างสะโพก | Euclidean(สะโพกซ้าย, สะโพกขวา) |
| 11 | `torso_length` | ความยาวลำตัว | Euclidean(ไหล่ซ้าย, สะโพกซ้าย) |
| **2 ความสมมาตร (Symmetry)** |||
| 12 | `elbow_symmetry` | ความสมมาตรข้อศอก | |มุมข้อศอกซ้าย − มุมข้อศอกขวา| |
| 13 | `knee_symmetry` | ความสมมาตรเข่า | |มุมเข่าซ้าย − มุมเข่าขวา| |

### 🤖 Classification — Binary Model

```
Features (13 ค่า) ──▶ RandomForest Classifier (deadlift_form.pkl)
                         ├── ✅ Correct (ท่าถูกต้อง) + % ความมั่นใจ
                         └── ❌ Incorrect (ท่าผิด) + "Check your form"
```

- **Model**: Random Forest Classifier
- ใช้ **SMOTE** เพื่อ Balance ข้อมูลก่อนเทรน
- **ผลลัพธ์**: Binary (Correct / Incorrect) พร้อมเปอร์เซ็นต์ความมั่นใจ

---

## 3. โมเดล Bench Press (เบนช์เพรส)

### 📐 Feature Extraction — สกัด 13 Features

โมเดล Bench Press ใช้ Features ชุดเดียวกันกับ Deadlift (**13 ค่า**) แต่โมเดลจะให้น้ำหนักความสำคัญแตกต่างกัน โดยเฉพาะมุมไหล่/ข้อศอก และความสมมาตรของแขน

| # | Feature | คำอธิบาย |
|---|---------|---------|
| 1–8 | **8 มุมข้อต่อ** | เหมือน Deadlift (elbow, shoulder, hip, knee ซ้าย/ขวา) |
| 9–11 | **3 ระยะทาง** | shoulder_width, hip_width, torso_length |
| 12–13 | **2 ความสมมาตร** | elbow_symmetry, knee_symmetry |

### 🤖 Classification — Voting Ensemble

```
Features (13 ค่า) ──▶ VotingClassifier (benchpress_form.pkl)
                         │   ├── XGBoost
                         │   └── RandomForest
                         │
                         ├── ✅ Correct (ท่าถูกต้อง)
                         └── ❌ Incorrect + "Keep elbows tucked, back arched, wrists straight"
```

- **Model**: Voting Classifier (Soft Voting) — **XGBoost + RandomForest**
- ใช้ **SMOTE** เพื่อ Balance ข้อมูลก่อนเทรน
- **ผลลัพธ์**: Binary (Correct / Incorrect) พร้อมคำแนะนำเฉพาะสำหรับ Bench Press

---

## ตารางเปรียบเทียบ 3 โมเดล

| | **Squat** | **Deadlift** | **Bench Press** |
|---|:---:|:---:|:---:|
| **จำนวน Features** | 20 (12 base + 8 engineered) | 13 | 13 |
| **Feature Extractor** | `SquatFeatureExtractor` (เฉพาะทาง) | `FeatureExtractor` (ทั่วไป) | `FeatureExtractor` (ทั่วไป) |
| **จุดเน้น Features** | มุมเข่า, สะโพก, ข้อเท้า, spine, lateral deviation | มุมข้อต่อทั้งตัว, ระยะทาง, สมมาตร | มุมข้อศอก/ไหล่, ระยะทาง, สมมาตร |
| **จำนวน Models** | 2 (Binary + Multi-class) | 1 (Binary) | 1 (Binary) |
| **Algorithm** | VotingClassifier 5 ตัว (XGB+LGB+Cat+RF+GB) | Random Forest | VotingClassifier 2 ตัว (XGB+RF) |
| **การจำแนก** | 6 ประเภท (Correct + 5 ข้อผิดพลาด) | 2 ประเภท (Correct / Incorrect) | 2 ประเภท (Correct / Incorrect) |
| **ประเภทข้อผิดพลาด** | Shallow, Forward Lean, Knees Caving, Heels Off, Asymmetric | ท่าไม่ถูกต้องทั่วไป | ข้อศอก, หลังแอ่น, ข้อมือ |
| **Data Balancing** | SMOTE | SMOTE | SMOTE |
| **ไฟล์โมเดล** | `squat_form.pkl` + `squat_form_detailed.pkl` | `deadlift_form.pkl` | `benchpress_form.pkl` |
