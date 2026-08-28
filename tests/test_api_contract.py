import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_squat_contract():
    # Payload as sent by camera/page.tsx
    payload = {
        "left_knee_angle": 120.5,
        "right_knee_angle": 120.5,
        "left_hip_angle": 110.0,
        "right_hip_angle": 110.0,
        "left_ankle_angle": 90.0,
        "right_ankle_angle": 90.0,
        "spine_angle": 45.0,
        "torso_lean": 45.0,
        "left_knee_lateral": 0.05,
        "right_knee_lateral": 0.05,
        "symmetry_score": 0.0,
        "hip_depth": 0.6
    }
    
    response = client.post("/predict/squat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "form_correct" in data
    assert "confidence" in data
    assert "feedback" in data

def test_deadlift_contract():
    # Payload as sent by camera/page.tsx for deadlift
    payload = {
        "features": [
            160.0, 160.0,  # elbows
            150.0, 150.0,  # shoulders
            120.0, 120.0,  # hips
            140.0, 140.0,  # knees
            0.5, 0.5, 0.4, # widths/length
            0.0, 0.0       # symmetry
        ]
    }
    
    response = client.post("/predict/deadlift", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "form_correct" in data
    assert "confidence" in data

def test_benchpress_contract():
    payload = {
        "features": [160]*13
    }
    response = client.post("/predict/benchpress", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "form_correct" in data
