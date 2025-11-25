import os
import mlflow
from mlflow.tracking import MlflowClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_mlflow():
    """
    Setup MLflow tracking dengan DagsHub
    """
    username = os.getenv("DAGSHUB_USERNAME")
    repo = os.getenv("DAGSHUB_REPO")
    token = os.getenv("DAGSHUB_TOKEN")
    
    if not all([username, repo]):
        print("[WARN] ⚠️  DagsHub credentials not found. Using local MLflow...")
        mlflow.set_tracking_uri("file:./mlruns")
        return False
    
    try:
        tracking_uri = f"https://dagshub.com/{username}/{repo}.mlflow"
        mlflow.set_tracking_uri(tracking_uri)

        os.environ['MLFLOW_TRACKING_USERNAME'] = username
        os.environ['MLFLOW_TRACKING_PASSWORD'] = token

        client = MlflowClient()
        experiments = client.search_experiments()  

        print(f"[INFO] Connected to DagsHub MLflow")
        print(f"[INFO] Tracking URI: {tracking_uri}")

        return True

    except Exception as e:
        print(f"[ERROR] DagsHub connection failed: {e}")
        print("[INFO] Falling back to local MLflow...")
        mlflow.set_tracking_uri("file:./mlruns")
        return False


def get_or_create_experiment(experiment_name):
    """
    Get atau create MLflow experiment
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"[INFO] Created experiment: {experiment_name}")
        else:
            experiment_id = experiment.experiment_id
            print(f"[INFO] Using existing experiment: {experiment_name}")
        
        mlflow.set_experiment(experiment_name)
        return experiment_id

    except Exception as e:
        print(f"[ERROR] Failed to setup experiment: {e}")
        raise

if __name__ == "__main__":
    print("=== TESTING MLFLOW CONFIG ===")
    setup_mlflow()
    get_or_create_experiment("test_experiment_config")
    print("=== DONE ===")
