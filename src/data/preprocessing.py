import pandas as pd
from src.data.ingestion import get_data
import logging
from pathlib import Path
logger=logging.getLogger(__name__)

def clean_data(input_path :Path,output_path :Path)-> pd.DataFrame:
    logger.info('Starting data preprocessing')
    df=get_data(input_path)
    df=df.copy()
    df=df.drop(columns=['Customer'])
    df['age']=pd.to_numeric(df['age'])
    df['zipcodeOri']=pd.to_numeric(df['zipcodeOri'])
    df['zipmerchant']=pd.to_numeric(df['zipmerchant'])
    df['merchant']=df['merchant'].str.strip("'")
    df['gender']=df['gender'].str.strip("'")
    logger.info('preprocessing completed')
    return df 
if __name__=='__main__' :
    input_path='../data/raw/data.csv'
    output_path='../data/processed/bank_cleaned_data'
    clean_data(input_path,output_path)