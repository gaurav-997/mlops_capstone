import pandas as pd
import re
import time
import logging
import string
pd.set_option('future.no_silent_downcasting', True)
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score , recall_score , f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download("wordnet", force=True)
nltk.download("omw-1.4", force=True)
nltk.download('stopwords')

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# setup mlflow & Dagshub
mlflow.set_tracking_uri(uri="https://dagshub.com/chauhan7gaurav/mlops_capstone.mlflow")
dagshub.init(repo_owner='chauhan7gaurav', repo_name='mlops_capstone', mlflow=True)
mlflow.set_experiment("BOW vs TFIDF experiment")

# data cleaning 

def lemmatization(text):
    """Lemmatize the text word by word."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def lower_case(text):
    return text.lower()

def remove_numbers(text):
    return "".join(char for char in text if not char.isdigit())


def removing_urls(text):
    """Remove URLs from the text."""
    URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
    return URL_PATTERN.sub('', text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = text.replace("؛", "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_stop_woords(text):
    stop_words = set(stopwords.words('english'))
    return " ".join(word for word in text.split() if word not in stop_words )
    
def normalize_text(df):
    try:
        df['review'] = df['review'].apply(removing_urls)
        df['review'] = df['review'].apply(removing_punctuations)
        df['review'] = df['review'].apply(remove_numbers)
        df['review'] = df['review'].apply(remove_stop_woords)
        df['review'] = df["review"].apply(lower_case)
        df["review"] = df["review"].apply(lemmatization)
        return df
    except Exception as e:
        raise e
  
# load data & cleaning   
def load_data(file_path):
    try:
        df = pd.read_csv(filepath_or_buffer="IMDB.csv")
        df = normalize_text(df)
        df['sentiment'] = df['sentiment'].map({'positive':1,'negative':0})
        return df
    except Exception as e:
        raise e
    
# feature engineering ( define vectoriser to use and algos )

VECTORIZERS = {
    'bow': CountVectorizer(),
    'tfidf': TfidfVectorizer()

}

ALGOS = {
    'LogisticRegression': LogisticRegression(),
    'MultinomialNB': MultinomialNB(),
    # 'XGBoost': XGBClassifier(),
    'RandomForest': RandomForestClassifier(),
    'GradientBoosting': GradientBoostingClassifier()
}

# train and evalutate models ( iterate over vectoriser and algos )

def train_evaluate(df):
    with mlflow.start_run(run_name="all runs") as parent_run:
        start_time = time.time()
        for vect_name,vectorizer in VECTORIZERS.items():
            for algo_name,algo in ALGOS.items():
                with mlflow.start_run(run_name=f"{vect_name} with {algo_name}", nested=True) as child_run:
                    try:
                        logging.info("divide the data ")
                        X = vectorizer.fit_transform(df['review'])
                        y = df['sentiment']
                        
                        logging.info('spliting data into train test ')
                        X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=42)
                        
                        logging.info("log processing params ")
                        mlflow.log_params({
                            'vectorizer':vect_name,
                            'algo':algo_name,
                            'test_size':0.2
                        })
                        
                        logging.info("initilizing the model and fiting it ")
                        model = algo
                        model.fit(X_train,y_train)
                        
                        logging.info("logging model params ")
                        log_model_params(algo_name,model)
                        
                        logging.info("making prediction on test data")
                        y_pred = model.predict(X_test)
                        
                        logging.info("calculating performance metrics ")
                        metrics = {
                            'accuracy' : accuracy_score(y_test,y_pred),
                            'precision': precision_score(y_test,y_pred),
                            'recall':recall_score(y_test,y_pred),
                            'f1': f1_score(y_test,y_pred)
                        }
                        
                        logging.info("logging metics and saving model")
                        mlflow.log_metrics(metrics)
                        mlflow.sklearn.log_model(model,"model")
                        
                        print(f"\nAlgorithm: {algo_name}, Vectorizer: {vect_name}") 
                        logging.info(metrics)
                        
                       
                        
                    except Exception as e:
                        logging.error(f"Failed for {vect_name} + {algo_name}: {e}")
                        mlflow.log_param("status", "failed")
                        mlflow.log_param("error", str(e))
                        continue

                    
def log_model_params(algo_name, model):
    """Logs hyperparameters of the trained model to MLflow."""
    params_to_log = {}
    if algo_name == 'LogisticRegression':
        params_to_log["C"] = model.C
    elif algo_name == 'MultinomialNB':
        params_to_log["alpha"] = model.alpha
    elif algo_name == 'XGBoost':
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["learning_rate"] = model.learning_rate
    elif algo_name == 'RandomForest':
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["max_depth"] = model.max_depth
    elif algo_name == 'GradientBoosting':
        params_to_log["n_estimators"] = model.n_estimators
        params_to_log["learning_rate"] = model.learning_rate
        params_to_log["max_depth"] = model.max_depth

    mlflow.log_params(params_to_log)


if __name__ == '__main__':
    df = load_data(file_path="IMDB.csv")
    train_evaluate(df)
                   


