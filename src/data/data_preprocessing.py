import pandas as pd
from src.logger import logging
import os
import string
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

nltk.download("wordnet", force=True)
nltk.download("omw-1.4", force=True)
nltk.download('stopwords')

def preprocess_dataframe(df:pd.DataFrame , col='text') -> pd.DataFrame:
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    def preprocess_text(text):
        #  Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        #  remove numbers 
        text = "".join([char for char in text if not char.isdigit()])
        
        text = text.lower()
        # Remove punctuations
        text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
        text = text.replace('؛', "")
        text = re.sub(r'\s+', ' ', text).strip()
        
        text =  "".join([word for word in text.split() if word not in stop_words])
        
         # Lemmatization
        text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
        return text
    
    df[col] = df[col].apply(preprocess_text)
    df = df.dropna(subset=[col])  # removing NA values 
    
    return df

def main():
    try:
        logging.info("loading raw data")
        train_data = pd.read_csv('./data/raw_data/train.csv')
        test_data = pd.read_csv('./data/raw_data/test.csv')
        
        logging.info("pre processing raw data ")
        processed_train_data = preprocess_dataframe(df=train_data ,col='review')
        processed_test_data = preprocess_dataframe(df= test_data ,col='review')
        
        logging.info("loading the processed data  ")
        processed_data_path = os.path.join('./data','interim')
        os.makedirs(processed_data_path,exist_ok=True)
        
        processed_train_data.to_csv(os.path.join(processed_data_path,"train_processed.csv"),index=False)
        processed_test_data.to_csv(os.path.join(processed_data_path,"test_processed.csv"),index=False)
        
        
        
    except Exception as e:
        raise e
    
if __name__ == "__main__":
    main()
        
