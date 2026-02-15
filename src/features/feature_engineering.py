import pandas as pd
import os
from src.logger import logging
from sklearn.feature_extraction.text import CountVectorizer
import pickle


def load_data(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
        logging.info("Replaces missing values with empty strings")
        df.fillna('', inplace=True)
        return df
    except Exception as e:
        raise e


def save_dataframe(df: pd.DataFrame, file_path: str) -> None:
    try:
        data_dir = os.path.dirname(file_path)
        os.makedirs(data_dir, exist_ok=True)
        df.to_csv(file_path, index=False)
    except Exception as e:
        raise e


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Move label encoding here (Feature Engineering stage)
    """
    # Normalize labels FIRST
    df['sentiment'] = df['sentiment'].astype(str).str.strip().str.lower()
    df = df[df['sentiment'].isin(['positive', 'negative'])].copy()
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0 }).astype(int)

    return df


def apply_bow(train_data: pd.DataFrame,test_data: pd.DataFrame,max_features: int) -> tuple:

    try:
        # label encoding moved here
        train_data = encode_labels(train_data)
        test_data = encode_labels(test_data)
        
        
        print(train_data['sentiment'].value_counts())
        print(test_data['sentiment'].value_counts())


        vectorizer = CountVectorizer(max_features=max_features)

        X_train = train_data['review'].values
        X_test = test_data['review'].values

        y_train = train_data['sentiment'].values
        y_test = test_data['sentiment'].values

        logging.info("converts text into Bag-of-Words vectors ")
        #  to avoide data leakage we are using BOW on training data 
        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)


        logging.info("convering above metrics into data frame ")
        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['sentiment'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['sentiment'] = y_test

        # ensure models dir exists
        os.makedirs("models", exist_ok=True)

        logging.info("saving model ")
        pickle.dump(vectorizer, open('models/vectorizer.pkl', 'wb'))

        logging.info('Bag of Words applied and data transformed')

        return train_df, test_df

    except Exception as e:
        raise e


def main():
    try:
        max_features = 2000

        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')
       

        train_df, test_df = apply_bow(train_data, test_data, max_features)

        save_dataframe(train_df, "./data/processed/train_bow.csv")
        save_dataframe(test_df, "./data/processed/test_bow.csv")

    except Exception as e:
        logging.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
