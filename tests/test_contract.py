import json
import pytest
from pathlib import Path
from app.schemas import SquatFeatures

def test_feature_contract():
    contract_path = Path(__file__).parent.parent / "app" / "feature_contract.json"
    assert contract_path.exists(), "feature_contract.json must exist"
    
    with open(contract_path, "r") as f:
        contract = json.load(f)
        
    # Check features13
    features13 = contract.get("features13", [])
    assert len(features13) == 13, "features13 must contain exactly 13 features"
    
    # Check features20 and SquatFeatures alignment
    features20 = contract.get("features20", [])
    assert len(features20) == 20, "features20 must contain exactly 20 features"
    
    # The first 12 features in features20 must match SquatFeatures fields in order
    schema_fields = list(SquatFeatures.model_fields.keys())
    assert len(schema_fields) == 12, "SquatFeatures must have exactly 12 fields"
    
    for i, field in enumerate(schema_fields):
        contract_feature_name = features20[i]["name"]
        assert contract_feature_name == field, f"Mismatch at index {i}: contract has '{contract_feature_name}', schema has '{field}'"
