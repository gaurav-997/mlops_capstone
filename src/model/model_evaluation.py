import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
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
mlflow.set_tracking_uri(uri="https://dagshub.com/chauhan7gaurav/mlops_capstone.mlflow")
dagshub.init(repo_owner='chauhan7gaurav', repo_name='mlops_capstone', mlflow=True)
mlflow.set_experiment("capston pipeline")

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info('Metrics saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the metrics: %s', e)
        raise