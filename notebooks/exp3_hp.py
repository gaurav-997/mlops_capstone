import os
import re
import string
import numpy as np
import pandas as pd
import mlflow
import logging
import mlflow.sklearn
import dagshub

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# =========================================================
# MLflow + Dagshub Setup
# =========================================================
mlflow.set_tracking_uri("https://dagshub.com/chauhan7gaurav/mlops_capstone.mlflow")
dagshub.init(repo_owner="chauhan7gaurav", repo_name="mlops_capstone", mlflow=True)
mlflow.set_experiment("Final Logistic regression params")

logging.basicConfig(level=logging.INFO)

# =========================================================
# Text Preprocessing
# =========================================================
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    text = " ".join(
        lemmatizer.lemmatize(word)
        for word in text.split()
        if word not in stop_words
    )
    return text.strip()


# =========================================================
# Load + Prepare Data
# =========================================================
def load_prepare_data(filepath):
    df = pd.read_csv(filepath)

    df["review"] = df["review"].astype(str).apply(preprocess_text)
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["review"])
    y = df["sentiment"]

    splits = train_test_split(X, y, test_size=0.2, random_state=42)
    return vectorizer, splits


# =========================================================
# Training + Logging
# =========================================================
def train_and_log_model(X_train, X_test, y_train, y_test, vectorizer):

    param_grid = {
        "C": [0.1, 1, 10],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],
    }

    # Parent Run
    with mlflow.start_run(run_name="LR_gridsearch_parent"):

        logging.info("Starting GridSearchCV...")

        grid_search = GridSearchCV(
            LogisticRegression(),
            param_grid,
            cv=5,
            scoring="f1",
            n_jobs=-1,
        )

        grid_search.fit(X_train, y_train)

        # =========================================
        # Child Runs (each hyperparameter combo)
        # =========================================
        for params, mean_score, std_score in zip(
            grid_search.cv_results_["params"],
            grid_search.cv_results_["mean_test_score"],
            grid_search.cv_results_["std_test_score"],
        ):

            with mlflow.start_run(
                run_name=f"LR params: {params}",
                nested=True,
            ):
                model = LogisticRegression(**params)
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1": f1_score(y_test, y_pred),
                    "mean_cv_score": mean_score,
                    "std_cv_score": std_score,
                }

                mlflow.log_params(params)
                mlflow.log_metrics(metrics)

                print(
                    f"Params: {params} | "
                    f"Accuracy: {metrics['accuracy']:.4f} | "
                    f"F1: {metrics['f1']:.4f}"
                )

        # =========================================
        # BEST MODEL Logging
        # =========================================
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_f1 = grid_search.best_score_

        logging.info("Logging best model and params")

        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1_cv", best_f1)

        mlflow.sklearn.log_model(best_model, "best_model")

        print(
            f"\nBEST MODEL: {best_model}\n"
            f"BEST PARAMS: {best_params}\n"
            f"BEST CV F1: {best_f1}"
        )


if __name__ == "__main__":
    vectorizer, (X_train, X_test, y_train, y_test) = load_prepare_data(filepath="IMDB.csv")
    train_and_log_model(X_train, X_test, y_train, y_test, vectorizer)
