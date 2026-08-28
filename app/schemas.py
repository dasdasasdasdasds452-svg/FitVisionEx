"""FitVision — Pydantic schemas"""
from pydantic import BaseModel, Field, model_validator
from typing import Optional
import math

class Features13(BaseModel):
    """13 features for deadlift / exercise classifier"""
    features: list[float] = Field(min_length=13, max_length=13)
    
    @model_validator(mode='after')
    def check_finite(self):
        for v in self.features:
            if not math.isfinite(v):
                raise ValueError(f"Feature values must be finite numbers, got {v}")
        return self

class SquatFeatures(BaseModel):
    """12 raw squat features — engineered features computed server-side"""
    left_knee_angle:    float
    right_knee_angle:   float
    left_hip_angle:     float
    right_hip_angle:    float
    left_ankle_angle:   float
    right_ankle_angle:  float
    spine_angle:        float
    torso_lean:         float
    left_knee_lateral:  float
    right_knee_lateral: float
    symmetry_score:     float
    hip_depth:          float

    @model_validator(mode='after')
    def check_finite(self):
        for k, v in self.model_dump().items():
            if not math.isfinite(v):
                raise ValueError(f"Squat feature '{k}' must be finite, got {v}")
        return self

class ExercisePrediction(BaseModel):
    exercise:   str
    confidence: float

class FormPrediction(BaseModel):
    form_correct:      bool
    confidence:        float
    feedback:          str
    error_type:        Optional[str] = None
    error_code:        Optional[int] = None
    detail_confidence: Optional[float] = None
