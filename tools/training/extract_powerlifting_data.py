import numpy as np
import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm

# Load the exact same feature extractor used in the frontend
sys.path.append('c:/fit/FitVision')
from src.features.squat_features import SquatFeatureExtractor

DATA_DIR = Path(r"C:\Users\tonkla\Downloads\archive (3)\Squat_Data\Squat_Data")
OUT_CSV = Path(r"c:\fit\FitVision\data\raw\powerlifting_squat\extracted_features.csv")
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

class DummyLandmark:
    """Mock MediaPipe landmark object for the extractor"""
    def __init__(self, x, y, z, vis):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = vis

class DummyPoseLandmarks:
    """Mock MediaPipe pose_landmarks array"""
    def __init__(self, npy_array):
        # npy_array is (132,) representing 33 landmarks * 4 coords (x,y,z,v)
        reshaped = npy_array.reshape(33, 4)
        self.landmark = [DummyLandmark(row[0], row[1], row[2], row[3]) for row in reshaped]

def extract_dataset():
    extractor = SquatFeatureExtractor()
    features_list = []
    
    # Valid = 0 (Correct)
    # Invalid = 1 (Incorrect - we will need to re-label these specifically later or just use as generic Incorrect)
    # For now, we extract everything
    
    for category in ["Valid", "Invalid"]:
        label = 0 if category == "Valid" else 1
        cat_dir = DATA_DIR / category
        
        # Structure is DATA_DIR / Category / Video_ID / frame_ID.npy
        for vid_dir in tqdm(list(cat_dir.iterdir()), desc=f"Processing {category}"):
            if not vid_dir.is_dir(): continue
            
            for file_path in vid_dir.glob("*.npy"):
                try:
                    data = np.load(file_path)
                    landmarks = DummyPoseLandmarks(data)
                    feat = extractor.extract_squat_features(landmarks)
                    
                    feat['video_id'] = f"{category}_{vid_dir.name}"
                    feat['frame_id'] = file_path.stem
                    feat['label'] = label
                    
                    features_list.append(feat)
                except Exception as e:
                    print(f"Error on {file_path}: {e}")
                    
    df = pd.DataFrame(features_list)
    print(f"\nExtracted {len(df)} frames.")
    print(f"Label dist:\n{df['label'].value_counts()}")
    
    df.to_csv(OUT_CSV, index=False)
    print(f"Saved to {OUT_CSV}")

if __name__ == "__main__":
    extract_dataset()
