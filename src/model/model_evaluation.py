import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score,f1_score, roc_auc_score
import pickle
import json
from src.logger import logging
import mlflow 
import mlflow.sklearn
import dagshub
import os

def load_data(file_path:str):
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise e
    
def load_model(file_path:str):
    try:
        with open(file_path,'rb') as file:
            model = pickle.load(file)
            
        return model
        
    except Exception as e:
        raise e
    
# setup mlflow & Dagshub

# Set up DagsHub credentials for MLflow tracking
# dagshub_token = os.getenv("CAPSTONE_TEST")
# if not dagshub_token:
#     raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com"
# repo_owner = "chauhan7gaurav"
# repo_name = "mlops_capstone"

# # Set up MLflow tracking URI
# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')


mlflow.set_tracking_uri(uri="https://dagshub.com/chauhan7gaurav/mlops_capstone.mlflow")
dagshub.init(repo_owner='chauhan7gaurav', repo_name='mlops_capstone', mlflow=True)
mlflow.set_experiment("capston pipeline")

def evaluate_model(clf , X_test , y_test ):
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred)
        auc = roc_auc_score(y_test,y_pred_proba)
        
        metrics = {
            'accuracy':accuracy,
            'precision': precision,
            'recall': recall,
            'f1':f1,
            'auc':auc
        }
        
        # mlflow.log_metrics(metrics)
        
        logging.info(f'Model evaluation metrics calculated: {metrics}')
        return metrics, y_pred, y_test
    
    except Exception as e:
        raise e

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info('Metrics saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the metrics: %s', e)
        raise
 
"""Save the model run ID and path to a JSON file."""   
def save_model_info(run_id:str , model_path:str , file_path:str ):
    try:
        model_info = {'run_id':run_id , 'model_path':model_path}
        with open(file_path,'w') as file:
            json.dump(model_info,file , indent=4)
    except Exception as e:
        raise e

def main():
    with mlflow.start_run() as run:
        try:
    
            test_data = load_data('./data/processed/test_bow.csv')

            X_test = test_data.iloc[:,:-1].values
            y_test = test_data.iloc[: ,-1].values

            clf = load_model(file_path="./models/model.pkl")

            metrics, y_pred, y_test = evaluate_model(clf , X_test , y_test)
            
            os.makedirs("reports", exist_ok=True)
            save_metrics(metrics , file_path='reports/metrics.json')

            logging.info(" logging metrics ")
            for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)

            logging.info(" logging model ")
            # use the correct parameter name: artifact_path (not name)
            mlflow.sklearn.log_model(clf, artifact_path="model")


            logging.info("saving model")
            save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')

            logging.info("logging artifacts")
            mlflow.log_artifact('reports/metrics.json')
            print("Unique predictions:", np.unique(y_pred, return_counts=True))
            print("Unique labels:", np.unique(y_test, return_counts=True))

        except Exception as e:
            raise e
        
if __name__ == '__main__':
    main()
        
                
        