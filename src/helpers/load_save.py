import os 
import yaml
import joblib
import json
import pandas as pd
import datetime

import pandas as pd 
from logging import Logger
from pandas import DataFrame
from typing import Any

## I/O
def load_data(file_path: str, logger : Logger) -> DataFrame:
    """Loads data from a csv file_path"""
    try : 
        df = pd.read_csv(file_path, engine='python')
        logger.debug("Data loaded from %s", file_path)
        return df 
    except pd.errors.ParserError as e: 
        logger.error("Failed to parse the CSV file : %s", e)
        raise
    except Exception as e: 
        logger.error("Unexpected error occured while loading the data : %s", e)
        raise


def save_data(train_data: DataFrame, test_data: DataFrame, data_path: str, logger : Logger) -> None: 
    try : 
        raw_data_path = os.path.join(data_path)
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)

        logger.debug("Train and test data saved to %s", raw_data_path)
    except Exception as e: 
        logger.error("Unexpected error occured while saving the data: %s", e)
        raise

def load_yaml(params_file_path : str, logger: Logger) -> dict:
    try: 
        with open(params_file_path, 'r') as file: 
            params = yaml.safe_load(file)
        logger.debug("Parameters retrieved from %s",params_file_path)
        return params
    except FileNotFoundError: 
        logger.error('File not found : %s', params_file_path)
        raise
    except yaml.YAMLError as e : 
        logger.error("YAML error : %s", e)
        raise
    except Exception as e : 
        logger.error("ERROR : %s", e)
        raise

def load_artifacts(artifact_path : str, logger: Logger) -> Any: 
    try : 
        artifact = joblib.load(artifact_path)
        logger.debug("Artifact '%s' loaded", artifact_path)
        return artifact
    except FileNotFoundError as e : 
        logger.error("File path not found : %s", e)
        raise
    except Exception as e: 
        logger.error("Unexpected Error occued while loading the artifact : %s",e )
        raise

def save_artifacts(artifact, artifact_path : str, logger: Logger) -> None: 
    try : 
        os.makedirs(os.path.dirname(artifact_path), exist_ok=True)

        with open(artifact_path, 'wb') as f: 
            joblib.dump(artifact, f)
        logger.debug("Artifact saved at %s", artifact_path)
    except FileNotFoundError as e : 
        logger.error("File path not found : %s", e)
        raise
    except Exception as e: 
        logger.error("Unexpected Error occued while saving the artifact : %s",e )
        raise

def log_dataset_profile(
    df: pd.DataFrame,
    dataset_name: str,
    output_path: str,
    bucket_name: str | None = None,
    object_key: str | None = None,
) -> None:
    """
    Generates a dataset profiling report and saves it as JSON.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to profile
    dataset_name : str
        Logical dataset name
    output_path : str
        Path where report should be saved
    bucket_name : str, optional
        Storage bucket name (MinIO/S3)
    object_key : str, optional
        Object path in storage
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    profile = {
        "dataset_name": dataset_name,
        "timestamp": str(datetime.datetime.now(datetime.UTC)),
        "storage_info": {
            "bucket_name": bucket_name,
            "object_key": object_key,
        },
        "shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "duplicates": int(df.duplicated().sum()),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024**2), 2),
        "columns": {},
    }

    for col in df.columns:
        profile["columns"][col] = {
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "null_percentage": round((df[col].isnull().mean()) * 100, 2),
            "unique_values": int(df[col].nunique()),
        }

    with open(output_path, "w") as f:
        json.dump(profile, f, indent=4)