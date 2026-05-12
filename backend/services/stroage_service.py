"""
backend/services/storage_service.py
AWS S3 service for DICOM storage and retrieval.
HIPAA-aligned: files encrypted at rest, access logged.
"""

import boto3
import uuid
import hashlib
import logging
from datetime import datetime, timezone
from typing import Optional, Dict
from botocore.exceptions import ClientError

from backend.config.settings import get_settings

logger = logging.getLogger(__name__)


class S3StorageService:
    """
    Manages DICOM and image storage on AWS S3.
    Uses server-side encryption (SSE-S3) by default.
    """

    def __init__(self):
        s = get_settings()
        self.bucket = s.s3_bucket_name
        self.region = s.aws_region

        boto_kwargs = {
            "region_name": s.aws_region,
        }
        if s.aws_access_key_id:
            boto_kwargs["aws_access_key_id"]     = s.aws_access_key_id
            boto_kwargs["aws_secret_access_key"] = s.aws_secret_access_key
        if s.aws_endpoint_url:
            boto_kwargs["endpoint_url"] = s.aws_endpoint_url

        self.s3 = boto3.client("s3", **boto_kwargs)

    def upload_dicom(
        self,
        file_bytes: bytes,
        modality: str,
        session_id: str,
        content_type: str = "application/dicom",
    ) -> Dict[str, str]:
        """
        Upload DICOM to S3 with encryption and metadata.
        Returns the S3 key and a content hash for verification.
        """
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        timestamp = datetime.now(timezone.utc).strftime("%Y/%m/%d")
        s3_key    = f"dicom/{modality}/{timestamp}/{session_id}/{file_hash[:16]}.dcm"

        try:
            self.s3.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=file_bytes,
                ContentType=content_type,
                ServerSideEncryption="AES256",  # SSE-S3
                Metadata={
                    "session-id": session_id,
                    "modality":   modality,
                    "upload-ts":  datetime.now(timezone.utc).isoformat(),
                    # NOTE: No patient identifiers stored (HIPAA)
                },
            )
            logger.info(f"Uploaded DICOM to s3://{self.bucket}/{s3_key}")
            return {"s3_key": s3_key, "content_hash": file_hash}

        except ClientError as e:
            logger.error(f"S3 upload failed: {e}")
            raise

    def get_presigned_url(self, s3_key: str, expiry_seconds: int = 3600) -> str:
        """Generate a presigned URL for temporary file access."""
        try:
            url = self.s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": s3_key},
                ExpiresIn=expiry_seconds,
            )
            return url
        except ClientError as e:
            logger.error(f"Presigned URL generation failed: {e}")
            raise

    def download(self, s3_key: str) -> bytes:
        """Download file bytes from S3."""
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
            return response["Body"].read()
        except ClientError as e:
            logger.error(f"S3 download failed: {e}")
            raise

    def delete(self, s3_key: str):
        """Delete file from S3 (for data retention compliance)."""
        try:
            self.s3.delete_object(Bucket=self.bucket, Key=s3_key)
            logger.info(f"Deleted s3://{self.bucket}/{s3_key}")
        except ClientError as e:
            logger.error(f"S3 delete failed: {e}")
            raise

    def list_session_files(self, session_id: str) -> list:
        """List all files associated with a session."""
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix=f"dicom/",
            )
            return [
                obj["Key"] for obj in response.get("Contents", [])
                if session_id in obj["Key"]
            ]
        except ClientError as e:
            logger.error(f"S3 list failed: {e}")
            return []