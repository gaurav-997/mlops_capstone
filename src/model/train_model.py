import pandas as pd
import numpy as np
from src.logger import logging
from sklearn.linear_model import LogisticRegression
import os
import pickle


def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise e


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    try:
        clf = LogisticRegression(C=1, solver='liblinear', penalty='l1')
        clf.fit(X_train, y_train)
        logging.info("Model training completed")
        return clf
    except Exception as e:
        raise e


def save_model(model, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logging.info(f"Model saved at {file_path}")
    except Exception as e:
        raise e


def main():
    try:
        train_data = load_data('./data/processed/train_bow.csv')

        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(X_train, y_train)

        save_model(clf, 'models/model.pkl')

    except Exception as e:
        logging.error("Model training failed: %s", e)
        raise e


if __name__ == '__main__':
    main()
