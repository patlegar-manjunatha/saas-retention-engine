import os
import sys
import pytest
import pandas as pd
from unittest.mock import patch, mock_open, MagicMock
from logging import Logger
import logging

from src.helpers.load_save import load_data, save_data, load_yaml, load_artifacts, save_artifacts
from src.helpers.exception_handling import MyException, error_message_detail
from src.helpers.initiate_stage import initiate_file, create_logger

@pytest.fixture
def mock_logger():
    """A dummy logger to pass into our functions so we don't clutter console output during tests."""
    return MagicMock(spec=Logger)

@pytest.fixture
def sample_df():
    """A tiny dataframe to use for testing I/O."""
    return pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})


@patch('src.helpers.load_save.pd.read_csv')
def test_load_data_success(mock_read_csv, mock_logger, sample_df):
    # Arrange: Tell the mock to return our sample DataFrame
    mock_read_csv.return_value = sample_df
    
    # Act
    result = load_data("dummy/path.csv", mock_logger)
    
    # Assert
    mock_read_csv.assert_called_once_with("dummy/path.csv", engine='python')
    pd.testing.assert_frame_equal(result, sample_df)
    mock_logger.debug.assert_called_once()

@patch('src.helpers.load_save.pd.read_csv')
def test_load_data_parser_error(mock_read_csv, mock_logger):
    # Arrange: Force the mock to raise an error
    mock_read_csv.side_effect = pd.errors.ParserError("Bad CSV")
    
    # Act & Assert: Verify that our function catches and re-raises it
    with pytest.raises(pd.errors.ParserError):
        load_data("dummy/path.csv", mock_logger)
    mock_logger.error.assert_called_once()

@patch('src.helpers.load_save.os.makedirs')
@patch('pandas.DataFrame.to_csv')
def test_save_data_success(mock_to_csv, mock_makedirs, mock_logger, sample_df):
    # Act
    save_data(sample_df, sample_df, "dummy_data_path", mock_logger)
    
    # Assert
    mock_makedirs.assert_called_once_with("dummy_data_path", exist_ok=True)
    assert mock_to_csv.call_count == 2  # Once for train, once for test
    mock_logger.debug.assert_called_once()

@patch('builtins.open', new_callable=mock_open, read_data="key: value")
def test_load_yaml_success(mock_file, mock_logger):
    # Act
    result = load_yaml("dummy.yaml", mock_logger)
    
    # Assert
    assert result == {"key": "value"}
    mock_logger.debug.assert_called_once()

@patch('builtins.open', side_effect=FileNotFoundError)
def test_load_yaml_not_found(mock_file, mock_logger):
    # Act & Assert
    with pytest.raises(FileNotFoundError):
        load_yaml("missing.yaml", mock_logger)
    mock_logger.error.assert_called_once()

@patch('src.helpers.load_save.joblib.load')
def test_load_artifacts_success(mock_joblib_load, mock_logger):
    # Arrange
    mock_joblib_load.return_value = "dummy_model"
    
    # Act (Assuming you fix the bug to return the artifact!)
    result = load_artifacts("model.pkl", mock_logger)
    
    # Assert
    mock_joblib_load.assert_called_once_with("model.pkl")
    mock_logger.debug.assert_called_once()

@patch('src.helpers.load_save.joblib.dump')
@patch('builtins.open', new_callable=mock_open)
@patch('src.helpers.load_save.os.makedirs')
def test_save_artifacts_success(mock_makedirs, mock_file, mock_joblib_dump, mock_logger):
    # Act
    save_artifacts("dummy_model", "dir/model.pkl", mock_logger)
    
    # Assert
    mock_makedirs.assert_called_once_with("dir", exist_ok=True)
    mock_file.assert_called_once_with("dir/model.pkl", 'wb')
    mock_joblib_dump.assert_called_once()


def test_my_exception_generation(mock_logger):
    # Arrange: Force a native Python error to get a real traceback (sys.exc_info)
    try:
        1 / 0
    except Exception as e:
        # Act
        custom_exc = MyException(e, sys, mock_logger)
        
        # Assert
        exc_str = str(custom_exc)
        assert "division by zero" in exc_str 
        assert "test_helpers.py" in exc_str  # Verifies it caught the right file
        mock_logger.error.assert_called_once()


@patch('src.helpers.initiate_stage.os.makedirs')
@patch('src.helpers.initiate_stage.logging.FileHandler')
def test_create_logger(mock_file_handler, mock_makedirs):
    # Act
    logger = create_logger("test_component")
    
    # Assert
    mock_makedirs.assert_called_once_with('logs', exist_ok=True)
    assert logger.name == "test_component"
    assert logger.level == logging.DEBUG
    # Reset handlers to avoid test pollution
    logger.handlers.clear()

@patch('src.helpers.initiate_stage.load_yaml')
@patch('src.helpers.initiate_stage.create_logger')
def test_initiate_file_success(mock_create_logger, mock_load_yaml, mock_logger):
    # Arrange
    mock_create_logger.return_value = mock_logger
    mock_load_yaml.return_value = {"data_ingestion": {"param1": "value1"}}
    
    # Act
    params, returned_logger = initiate_file("params.yaml", "data_ingestion")
    
    # Assert
    assert params == {"param1": "value1"}
    assert returned_logger == mock_logger
    mock_load_yaml.assert_called_once_with("params.yaml", mock_logger)

@patch('src.helpers.initiate_stage.load_yaml')
@patch('src.helpers.initiate_stage.create_logger')
def test_initiate_file_key_error(mock_create_logger, mock_load_yaml, mock_logger):
    # Arrange
    mock_create_logger.return_value = mock_logger
    mock_load_yaml.return_value = {"other_component": {}} # Missing the component we ask for
    
    # Act & Assert
    with pytest.raises(KeyError):
        initiate_file("params.yaml", "data_ingestion")
    mock_logger.error.assert_called_once()