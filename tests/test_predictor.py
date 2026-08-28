import pytest
from app.predictor import ModelRegistry
from config.settings import MODELS_DIR

@pytest.fixture
def registry():
    return ModelRegistry(MODELS_DIR)

def test_squat_rules():
    # Shallow squat rule
    features = {"left_knee_angle": 140, "right_knee_angle": 140}
    from app.predictor import _classify_squat_error_rules
    assert _classify_squat_error_rules(features) == 1

    # Knees caving in
    features = {"left_knee_angle": 90, "right_knee_angle": 90, "left_knee_lateral": 0.1}
    assert _classify_squat_error_rules(features) == 3

    # Asymmetric
    features = {"left_knee_angle": 90, "right_knee_angle": 90, "symmetry_score": 80}
    assert _classify_squat_error_rules(features) == 5

def test_engineer_squat_features():
    from app.predictor import engineer_squat_features
    f = {"left_knee_angle": 90, "right_knee_angle": 90, "left_hip_angle": 90, "right_hip_angle": 90}
    vec = engineer_squat_features(f)
    assert len(vec) == 20
    assert vec[12] == 90  # avg_knee

def test_registry_lazy_loading(registry):
    # Should start empty
    assert len(registry._active_models) == 0
    
    # Load classifier
    models = registry._get_models("classifier")
    assert "exercise" in models
    assert len(registry._active_models) == 1
    
    # Load deadlift, should evict classifier
    models = registry._get_models("deadlift")
    assert "deadlift_form" in models
    assert "exercise" not in registry._active_models
