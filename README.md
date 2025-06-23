# Data-Internships
etl_pipeline/
├── etl_pipeline.py
├── config.py
├── logger.py
├── __init__.py
└── tests/
    ├── test_etl_pipeline.py
    └── conftest.py
pandas==2.2.2
scikit-learn==1.5.0
pytest==8.2.1
pip install -r requirements.txt
# etl_pipeline.py

"""
ETL pipeline for processing social advertisement data.
"""

import logging
from typing import Tuple
import pandas as pd
from sklearn.preprocessing import StandardScaler
from logger import get_logger

logger = get_logger(__name__)


class ETLPipeline:
    """A simple ETL pipeline for CSV data."""

    def __init__(self, input_path: str, output_path: str):
        """
        Initialize the ETL pipeline.
        
        Args:
            input_path (str): Path to the input CSV file.
            output_path (str): Path where transformed CSV will be saved.
        """
        self.input_path = input_path
        self.output_path = output_path

    def extract(self) -> pd.DataFrame:
        """
        Extract data from the CSV file.

        Returns:
            pd.DataFrame: Extracted data.
        """
        try:
            df = pd.read_csv(self.input_path)
            logger.info(f"Extracted {df.shape[0]} records from {self.input_path}")
            return df
        except Exception as e:
            logger.exception("Failed to extract data.")
            raise

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the dataset (e.g., scale numeric features).

        Args:
            df (pd.DataFrame): Raw dataframe.

        Returns:
            pd.DataFrame: Transformed dataframe.
        """
        try:
            if df.isnull().sum().any():
                logger.warning("Missing values found. Filling with median.")
                df.fillna(df.median(numeric_only=True), inplace=True)

            scaler = StandardScaler()
            df[['Age', 'EstimatedSalary']] = scaler.fit_transform(df[['Age', 'EstimatedSalary']])
            logger.info("Successfully transformed the dataset.")
            return df
        except Exception as e:
            logger.exception("Failed during transformation.")
            raise

    def load(self, df: pd.DataFrame) -> None:
        """
        Load the transformed data to a new CSV file.

        Args:
            df (pd.DataFrame): Transformed dataframe.
        """
        try:
            df.to_csv(self.output_path, index=False)
            logger.info(f"Transformed data saved to {self.output_path}")
        except Exception as e:
            logger.exception("Failed to load data.")
            raise

    def run(self) -> None:
        """
        Execute the ETL pipeline.
        """
        logger.info("Starting ETL pipeline...")
        df = self.extract()
        df_transformed = self.transform(df)
        self.load(df_transformed)
        logger.info("ETL pipeline completed successfully.")
# logger.py

"""
Centralized logging configuration.
"""

import logging


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger.
    
    Args:
        name (str): Logger name.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
# tests/test_etl_pipeline.py

import pytest
import pandas as pd
from etl_pipeline import ETLPipeline

@pytest.fixture
def dummy_data(tmp_path):
    file_path = tmp_path / "test_data.csv"
    df = pd.DataFrame({
        "Age": [25, 30, 35],
        "EstimatedSalary": [50000, 60000, 70000],
        "Purchased": [0, 1, 0]
    })
    df.to_csv(file_path, index=False)
    return file_path

def test_etl_pipeline_end_to_end(tmp_path, dummy_data):
    output_file = tmp_path / "output.csv"
    etl = ETLPipeline(str(dummy_data), str(output_file))
    
    df_raw = etl.extract()
    assert df_raw.shape == (3, 3)

    df_transformed = etl.transform(df_raw.copy())
    assert df_transformed.isnull().sum().sum() == 0

    etl.load(df_transformed)
    df_loaded = pd.read_csv(output_file)
    assert df_loaded.shape == (3, 3)
python -m etl_pipeline
