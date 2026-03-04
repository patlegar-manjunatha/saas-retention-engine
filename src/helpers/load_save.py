import os 
import yaml
import joblib

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