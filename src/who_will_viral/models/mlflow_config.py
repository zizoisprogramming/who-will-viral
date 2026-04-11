from pathlib import Path
import mlflow

ROOT     = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / 'data' / 'youtube'

MLFLOW_TRACKING_URI = (ROOT / 'mlruns').as_uri() 
EXPERIMENT_NAME     = 'who_will_viral'


def setup_mlflow():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)