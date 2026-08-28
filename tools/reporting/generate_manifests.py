import json
import hashlib
from pathlib import Path
from datetime import datetime

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "models"

def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def generate_manifest(model_name: str, features: list[str], description: str):
    pkl_path = MODELS_DIR / f"{model_name}.pkl"
    manifest_path = MODELS_DIR / f"{model_name}.manifest.json"
    
    if not pkl_path.exists():
        print(f"Skipping {model_name}, .pkl not found")
        return
        
    manifest = {
        "model_name": model_name,
        "description": description,
        "features": features,
        "created_at": datetime.now().isoformat(),
        "sha256": compute_sha256(pkl_path)
    }
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)
    print(f"Generated manifest for {model_name}")

if __name__ == "__main__":
    raw_13 = [f"feature_{i}" for i in range(13)]
    
    squat_features = [
        "left_knee_angle", "right_knee_angle", "left_hip_angle", "right_hip_angle",
        "left_ankle_angle", "right_ankle_angle", "spine_angle", "torso_lean",
        "left_knee_lateral", "right_knee_lateral", "symmetry_score", "hip_depth"
    ]
    
    generate_manifest("deadlift_form", raw_13, "Deadlift Form Classification (RF+SMOTE)")
    generate_manifest("benchpress_form", raw_13, "Benchpress Form Classification (XGBoost)")
    generate_manifest("exercise_classifier", raw_13, "Exercise Type Classification")
    generate_manifest("squat_form_3class", squat_features, "Squat 3-Class RF Model")
