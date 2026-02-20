import json
import mlflow
from src.logger import logging
import os
import warnings

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")


# ---------------------------
# MLflow Setup
# ---------------------------

def setup_mlflow():
    dagshub_token = os.getenv("CAPSTONE_TEST")
    if not dagshub_token:
        raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    mlflow.set_tracking_uri(
        "https://dagshub.com/chauhan7gaurav/mlops_capstone.mlflow"
    )


# ---------------------------
# Helpers
# ---------------------------

def load_model_info(file_path: str) -> dict:
    try:
        with open(file_path, "r") as file:
            model_info = json.load(file)
        logging.debug("Model info loaded from %s", file_path)
        return model_info
    except Exception as e:
        logging.error("Error loading model info: %s", e)
        raise


def register_model(model_name: str, model_info: dict):

    run_id = model_info["run_id"]
    requested_path = model_info.get("model_path", "")

    client = mlflow.tracking.MlflowClient()

    model_uri = f"runs:/{run_id}/{requested_path}"

    logging.info(f"Registering model from {model_uri}")

    model_version = mlflow.register_model(model_uri, model_name)

    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Staging",
    )

    logging.info(
        f"Model {model_name} version {model_version.version} moved to Staging"
    )


# ---------------------------
# Main
# ---------------------------

def main():
    try:
        setup_mlflow()

        model_info = load_model_info("reports/experiment_info.json")

        model_name = "my_model"
        register_model(model_name, model_info)

    except Exception as e:
        logging.error("Model registration failed: %s", e)
        raise


if __name__ == "__main__":
    main()