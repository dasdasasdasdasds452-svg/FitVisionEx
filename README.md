# FitVision — AI Exercise Form Analyzer

Real-time exercise form analysis using computer vision and machine learning. The system uses MediaPipe Pose to detect body landmarks from a webcam feed, extracts biomechanical features (joint angles, symmetry, depth), and predicts form correctness using trained ML models.

## Supported Exercises

| Exercise | Model Type | Classes | F1-macro |
|----------|-----------|---------|----------|
| **Squat** | XGBoost + Random Forest (Voting) | Good / Bad Back / Bad Heel | 0.88 |
| **Deadlift** | XGBoost + Random Forest (Voting) | Correct / Incorrect | 0.80 |
| **Bench Press** | XGBoost + Random Forest (Voting) | Correct / Incorrect | 0.99* |

*\*Bench press accuracy is inflated due to limited video diversity in training data.*

## Tech Stack

- **Frontend**: Next.js 16, React 19, TypeScript, MediaPipe Pose
- **Backend**: Python 3.11, FastAPI, scikit-learn, XGBoost
- **ML**: Ensemble (VotingClassifier: XGBoost + RandomForest), Isotonic Calibration
- **Deploy**: Vercel (Frontend), Railway/Koyeb (Backend)

## Project Structure

```
FitVision/           ← Python backend (FastAPI)
├── app/
│   ├── main.py      ← API endpoints
│   ├── predictor.py ← ML model serving (ModelRegistry)
│   └── schemas.py   ← Pydantic request/response schemas
├── config/
│   └── settings.py  ← Model paths, thresholds
├── data/
│   ├── models/      ← Deployed .pkl models
│   └── processed/   ← Training datasets
├── tools/
│   ├── training/    ← Model training scripts
│   └── evaluation/  ← Model evaluation scripts
├── tests/           ← Automated tests
├── Dockerfile       ← Multi-stage build with SHA256 verification
└── requirements-fly.txt

fitvision-next/      ← Next.js frontend
├── src/app/
│   ├── camera/      ← Real-time pose analysis page
│   ├── chat/        ← AI fitness chat
│   ├── api/ai/      ← Server-side AI routes
│   └── ...          ← Dashboard, history, settings
└── package.json
```

## Quick Start

### Backend
```bash
cd FitVision
pip install -r requirements-fly.txt
uvicorn app.main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

### Frontend
```bash
cd fitvision-next
cp .env.example .env.local  # Then edit with your API URL
npm install
npm run dev
# → http://localhost:3000
```

### Environment Variables

**Frontend** (`.env.local`):
```
NEXT_PUBLIC_API_URL=http://localhost:8000   # Backend URL
AI_API_KEY=sk_...                          # AI chat API key (server-only)
AI_BASE_URL=https://...                    # AI gateway URL (server-only)
AI_MODEL=gemini-2.5-flash                  # AI model name
```

**Backend** (env vars):
```
ALLOWED_ORIGINS=http://localhost:3000       # CORS origins (comma-separated)
SENTRY_DSN=...                             # Optional: error tracking
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict/squat` | Squat form prediction (12 named features) |
| `POST` | `/predict/deadlift` | Deadlift form prediction (13 features) |
| `POST` | `/predict/benchpress` | Bench press form prediction (13 features) |
| `POST` | `/predict/exercise` | Exercise type classifier |
| `POST` | `/report` | Submit prediction report (data flywheel) |
| `GET` | `/health` | Health check + model list |

## How It Works

1. **Camera** captures webcam frames in the browser
2. **MediaPipe Pose** detects 33 body landmarks per frame
3. **Feature extraction** computes joint angles, symmetry scores, depth ratios
4. **Features sent** to backend API every 5th frame
5. **ML model** predicts form correctness + error type
6. **Result displayed** in real-time with feedback overlay

## Authors

- FitVision Team — Computer Engineering, Khon Kaen University
