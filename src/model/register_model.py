# register model

import json
import mlflow
import logging
from src.logger import logging
import os
import dagshub

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "chauhan7gaurav"
repo_name = "mlops_capstone"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------


# Below code block is for local use
# -------------------------------------------------------------------------------------
# mlflow.set_tracking_uri(uri="https://dagshub.com/chauhan7gaurav/mlops_capstone.mlflow")
# dagshub.init(repo_owner='chauhan7gaurav', repo_name='mlops_capstone', mlflow=True)



def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        run_id = model_info['run_id']
        requested_path = model_info.get('model_path', '')

        client = mlflow.tracking.MlflowClient()

        # Try the expected path first
        tried_uris = []
        if requested_path:
            model_uri = f"runs:/{run_id}/{requested_path}"
            tried_uris.append(model_uri)
            try:
                model_version = mlflow.register_model(model_uri, model_name)
            except Exception:
                model_version = None
        else:
            model_version = None

        # If the simple approach failed, try to locate artifacts via the run's artifact_uri
        if model_version is None:
            try:
                run_info = client.get_run(run_id).info
                artifact_base = run_info.artifact_uri.rstrip('/')
            except Exception:
                artifact_base = None

            tried_candidates = []
            if artifact_base:
                # common candidate locations
                candidates = [
                    f"{artifact_base}/model",
                    f"{artifact_base}/{requested_path}",
                    f"{artifact_base}/model/model.pkl",
                    f"{artifact_base}/model.pkl",
                    f"{artifact_base}/{requested_path}/model.pkl",
                ]
                for c in candidates:
                    tried_candidates.append(c)
                    try:
                        model_version = mlflow.register_model(c, model_name)
                        model_uri = c
                        break
                    except Exception:
                        model_version = None

            # Last resort: try listing artifacts remotely and pick a candidate
            if model_version is None:
                artifacts = client.list_artifacts(run_id)
                candidate = None
                for a in artifacts:
                    if a.path == requested_path:
                        candidate = a.path
                        break
                if candidate is None:
                    for a in artifacts:
                        if a.is_dir:
                            candidate = a.path
                            break
                if candidate is None:
                    for a in artifacts:
                        if a.path.lower().endswith('.pkl') or a.path.lower().endswith('.joblib'):
                            candidate = a.path
                            break

                if candidate:
                    model_uri = f"runs:/{run_id}/{candidate}"
                    tried_uris.append(model_uri)
                    model_version = mlflow.register_model(model_uri, model_name)
                else:
                    all_tried = tried_uris + tried_candidates
                    logging.error('No suitable model artifact found for run %s. Tried URIs: %s', run_id, all_tried)
                    raise RuntimeError(f'Unable to find a model artifact for run {run_id}')

        # Transition the model to "Staging" stage
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        logging.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)
        
        model_name = "my_model"
        register_model(model_name, model_info)
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
