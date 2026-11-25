
import pandas as pd
import logging
from pathlib import Path
from src.data.ingestion import get_data

logger = logging.getLogger(__name__)


def clean_data(input_path: Path, output_path: Path) -> pd.DataFrame:

    logger.info("Starting data preprocessing")

    # Load data
    df = get_data(input_path)

    # Work on a copy
    df = df.copy()

    df = df.drop(columns=["Customer"])

    numerical_columns = ["age", "zipcodeOri", "zipMerchant"]

    for column in numerical_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    categorical_columns = ["merchant", "gender"]

    for column in categorical_columns:
        df[column] = df[column].astype("string").str.strip("'").str.strip()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info("Preprocessing completed")

    return df


if __name__ == "__main__":

    input_path = Path("../data/raw/data.csv")
    output_path = Path("../data/processed/bank_cleaned_data.csv")

    clean_data(input_path, output_path)
