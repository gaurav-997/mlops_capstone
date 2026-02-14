import pandas as pd

from src.logger import logging
import os
from sklearn.model_selection import train_test_split


def load_data(file_url) ->pd.DataFrame:
    try:
        logging.info(f"fetching data from {file_url}")
        df = pd.read_csv(file_url)
        return df
    except Exception as e:
        raise e
    
def validate_data(df:pd.DataFrame) -> pd.DataFrame:
    try:
        logging.info("validatine above loaded data ")
        required_columns = ["review" , "sentiment"]
        for column in required_columns:
            if column not in df.columns:
                raise ValueError(f"missing required columns ") 
        
        return df
    except Exception as e:
        raise e
    
def split_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        logging.info("spliting data into train and test ")
        train_data , test_data = train_test_split(df , test_size=0.2 ,random_state=42)
        return train_data , test_data
    except Exception as e:
        raise e
    
def save_data(train_data :pd.DataFrame , test_data: pd.DataFrame , dir_path: str) -> pd.DataFrame:
    try:
        logging.info("saving splited data ")
        raw_data_path = os.path.join(dir_path , "raw_data")
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,"train.csv"),index=False)
        test_data.to_csv(os.path.join(raw_data_path,"test.csv"),index=False)
        logging.info(f"train and test data saved in {raw_data_path}")
    except Exception as e:
        raise e
    
def main():
    try:
        
        df = load_data(file_url= "https://raw.githubusercontent.com/gaurav-997/mlops_capstone/main/notebooks/data.csv")
        validated_data = validate_data(df)
        train_data , test_data  = split_data(df=validated_data)
        save_data(train_data , test_data , dir_path= './data')
        
    except Exception as e:
        raise e
    
if __name__ == "__main__":
    main()
    