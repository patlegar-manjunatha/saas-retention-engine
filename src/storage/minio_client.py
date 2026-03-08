import os
from pathlib import Path
import boto3
from botocore.exceptions import ClientError


class MinIOClient:
    """
    Wrapper around S3 compatible storage (MinIO / AWS S3)

    Responsibilities:
    - Connect to object storage
    - Upload files
    - Download files
    - List objects
    - Ensure bucket exists
    """

    def __init__(
        self,
        endpoint_url: str,
        access_key: str,
        secret_key: str,
        bucket_name: str,
        region_name: str = "us-east-1",
    ):
        self.endpoint_url = endpoint_url
        self.bucket_name = bucket_name

        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint_url,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region_name,
        )

        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        """
        Creates bucket if it doesn't exist
        """
        try:
            self.client.head_bucket(Bucket=self.bucket_name)
        except ClientError:
            self.client.create_bucket(Bucket=self.bucket_name)


    def upload_file(self, local_path: str, object_key: str):
        """
        Upload file to object storage

        local_path  -> path on local system
        object_key  -> path inside bucket
        """
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"{local_path} not found")

        self.client.upload_file(
            Filename=local_path,
            Bucket=self.bucket_name,
            Key=object_key,
        )

    def download_file(self, object_key: str, local_path: str):
        """
        Download object from storage

        object_key -> path inside bucket
        local_path -> local save path
        """

        Path(local_path).parent.mkdir(parents=True, exist_ok=True)

        self.client.download_file(
            Bucket=self.bucket_name,
            Key=object_key,
            Filename=local_path,
        )

    def object_exists(self, object_key: str) -> bool:
        """
        Check if object exists in bucket
        """

        try:
            self.client.head_object(
                Bucket=self.bucket_name,
                Key=object_key,
            )
            return True
        except ClientError:
            return False


    def list_objects(self, prefix: str = ""):
        """
        List objects in bucket
        """

        response = self.client.list_objects_v2(
            Bucket=self.bucket_name,
            Prefix=prefix,
        )

        objects = []

        if "Contents" in response:
            for obj in response["Contents"]:
                objects.append(obj["Key"])

        return objects


    def delete_object(self, object_key: str):
        """
        Delete object from bucket
        """

        self.client.delete_object(
            Bucket=self.bucket_name,
            Key=object_key,
        )