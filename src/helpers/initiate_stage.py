import logging
import os 
from logging import Logger
from typing import Tuple

from src.helpers.load_save import load_yaml

def initiate_file(params_file_path : str, component_name : str) -> Tuple[dict, Logger]:
    """
Initialize stage-specific configuration and logger.

This function:
- Loads the YAML configuration file
- Extracts parameters for the given pipeline component
- Creates and returns a dedicated logger for that component

Args:
    params_file_path (str): Path to the YAML configuration file.
    component_name (str): Name of the pipeline component
                          (e.g., 'data_ingestion', 'data_preprocessing').

Returns:
    Tuple[dict, Logger]:
        - Dictionary containing configuration for the specified component
        - Logger instance scoped to the specified component

Raises:
    FileNotFoundError: If the params file does not exist.
    yaml.YAMLError: If the YAML file is malformed.
    KeyError: If the component name is not found in the configuration.
"""


    logger = create_logger(component_name)
    try : 
        params = load_yaml(params_file_path, logger)[component_name]
        
        return params, logger
    except KeyError as e : 
        logger.error('COMPONENT NAME NOT FOUND %s', e)
        raise
    

def create_logger(component_name : str) -> Logger:
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)

    filename = f"{component_name}.log"
    log_file_path = os.path.join(log_dir, filename)

    logger = logging.getLogger(component_name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger


