import os
import pytest
from unittest.mock import patch, MagicMock
from src.storage.minio_client import MinIOClient
from botocore.exceptions import ClientError

@pytest.fixture
def mock_s3_client():
    with patch("src.storage.minio_client.boto3.client") as mock_client:
        yield mock_client


@pytest.fixture
def minio_client(mock_s3_client):
    client_instance = MagicMock()
    mock_s3_client.return_value = client_instance

    return MinIOClient(
        endpoint_url="http://localhost:9000",
        access_key="admin",
        secret_key="admin123",
        bucket_name="test-bucket",
    )


def test_bucket_creation(mock_s3_client):
    client_instance = MagicMock()
    mock_s3_client.return_value = client_instance

    MinIOClient(
        endpoint_url="http://localhost:9000",
        access_key="admin",
        secret_key="admin123",
        bucket_name="test-bucket",
    )

    client_instance.head_bucket.assert_called_with(Bucket="test-bucket")



def test_upload_file(minio_client, tmp_path):

    test_file = tmp_path / "file.txt"
    test_file.write_text("test data")

    minio_client.upload_file(
        local_path=str(test_file),
        object_key="raw/file.txt",
    )

    minio_client.client.upload_file.assert_called_once()



def test_upload_file_missing(minio_client):

    with pytest.raises(FileNotFoundError):
        minio_client.upload_file(
            local_path="missing_file.txt",
            object_key="raw/file.txt",
        )


def test_download_file(minio_client, tmp_path):

    download_path = tmp_path / "download.txt"

    minio_client.download_file(
        object_key="raw/file.txt",
        local_path=str(download_path),
    )

    minio_client.client.download_file.assert_called_once()



def test_object_exists_true(minio_client):

    minio_client.client.head_object.return_value = {}

    result = minio_client.object_exists("raw/file.txt")

    assert result is True


def test_object_exists_false(minio_client):

    minio_client.client.head_object.side_effect = ClientError(
        error_response={"Error": {"Code": "404"}},
        operation_name="HeadObject",
    )

    result = minio_client.object_exists("raw/file.txt")

    assert result is False


def test_list_objects(minio_client):

    minio_client.client.list_objects_v2.return_value = {
        "Contents": [
            {"Key": "raw/file1.csv"},
            {"Key": "raw/file2.csv"},
        ]
    }

    objects = minio_client.list_objects(prefix="raw")

    assert objects == ["raw/file1.csv", "raw/file2.csv"]

def test_delete_object(minio_client):

    minio_client.delete_object("raw/file.txt")

    minio_client.client.delete_object.assert_called_once_with(
        Bucket="test-bucket",
        Key="raw/file.txt",
    )