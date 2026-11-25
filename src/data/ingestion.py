import logging 
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='$(asctime)s-%(levelname)s-%(message)s'
)

logger=logging.getLogger(__name__)

def get_data( file_path: Path)-> pd.DataFrame :
    logger.info('starting data ingestion')
    if not file_path.exists():
        raise FileNotFoundError('file not found')
    logger.info('loading dataset',file_path)
    df=pd.read_csv(file_path)
    logger.info('dataset laoded succefully')
    return df