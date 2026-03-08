import pytest
from unittest.mock import MagicMock, patch
from src.components.data_ingestion import DataIngestion
from src.helpers.exception_handling import MyException


@pytest.fixture
def mock_params():
    return {
        "bucket_name": "test-bucket",
        "object_key": "raw/test.csv",
        "local_path": "data/raw/test.csv",
        "output_profile_path": "artifacts/profile.json",
    }


@pytest.fixture
def mock_logger():
    return MagicMock()


@patch("src.components.data_ingestion.MinIOClient")
def test_init_success(mock_minio, mock_params, mock_logger):

    ingestion = DataIngestion(params=mock_params, logger=mock_logger)

    assert ingestion.object_key == "raw/test.csv"
    assert ingestion.local_path == "data/raw/test.csv"



@patch("src.components.data_ingestion.MinIOClient")
def test_download_file_success(mock_minio, mock_params, mock_logger):

    mock_client = MagicMock()
    mock_client.object_exists.return_value = True
    mock_minio.return_value = mock_client

    ingestion = DataIngestion(params=mock_params, logger=mock_logger)

    result = ingestion.download_file()

    mock_client.download_file.assert_called_once()
    assert result == "data/raw/test.csv"


@patch("src.components.data_ingestion.MinIOClient")
def test_download_file_missing(mock_minio, mock_params, mock_logger):

    mock_client = MagicMock()
    mock_client.object_exists.return_value = False
    mock_minio.return_value = mock_client

    ingestion = DataIngestion(params=mock_params, logger=mock_logger)

    with pytest.raises(MyException):
        ingestion.download_file()


@patch("src.components.data_ingestion.log_dataset_profile")
@patch("src.components.data_ingestion.load_data")
@patch("src.components.data_ingestion.MinIOClient")
@patch("src.components.data_ingestion.os.makedirs")
def test_run_success(
    mock_makedirs,
    mock_minio,
    mock_load_data,
    mock_profile,
    mock_params,
    mock_logger,
):

    mock_client = MagicMock()
    mock_client.object_exists.return_value = True
    mock_minio.return_value = mock_client

    mock_df = MagicMock()
    mock_load_data.return_value = mock_df

    ingestion = DataIngestion(params=mock_params, logger=mock_logger)

    ingestion.run()

    mock_load_data.assert_called_once()
    mock_profile.assert_called_once()



@patch("src.components.data_ingestion.MinIOClient")
def test_run_failure(mock_minio, mock_params, mock_logger):

    mock_client = MagicMock()
    mock_client.object_exists.side_effect = Exception("Storage failure")
    mock_minio.return_value = mock_client

    ingestion = DataIngestion(params=mock_params, logger=mock_logger)

    with pytest.raises(MyException):
        ingestion.run()