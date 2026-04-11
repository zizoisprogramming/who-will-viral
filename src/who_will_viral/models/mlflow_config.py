from pathlib import Path
import mlflow

BASE_DIR = Path(__file__).resolve().parent  # models/

MLFLOW_TRACKING_URI = (BASE_DIR / "mlruns").as_uri()
EXPERIMENT_NAME = "youtube-viral"

def setup_mlflow():
    (BASE_DIR / "mlruns").mkdir(exist_ok=True)  # ensure folder exists
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)