import mlflow
from typing import Dict, Any

class MLflowTracker:
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        
    def __enter__(self):
        self.run = mlflow.start_run()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        mlflow.end_run()
        
    def log_params(self, params: Dict[str, Any]):
        mlflow.log_params(params)
        
    def log_metrics(self, metrics: Dict[str, float]):
        mlflow.log_metrics(metrics)
        
    def log_model(self, model, artifact_path: str):
        try:
            # Simple fallback for sklearn models
            mlflow.sklearn.log_model(model, artifact_path)
        except Exception:
            pass
            
    def log_artifact(self, local_path: str):
        mlflow.log_artifact(local_path)
