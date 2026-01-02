"""
MinIO Chart Storage Client

Handles uploading and managing Claude vision analysis charts in MinIO object storage.
Features:
- Automatic bucket creation with 30-day lifecycle policy
- Public URL generation for Streamlit access
- Graceful fallback if MinIO is unavailable
"""

import logging
import os
import json
from io import BytesIO
from datetime import timedelta
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from minio import Minio
    from minio.error import S3Error
    from minio.lifecycleconfig import LifecycleConfig, Rule, Expiration
    MINIO_SDK_AVAILABLE = True
except ImportError:
    MINIO_SDK_AVAILABLE = False
    Minio = None
    S3Error = None
    logger.warning("MinIO SDK not installed. Run: pip install minio")


class MinIOChartClient:
    """
    MinIO client for storing Claude vision analysis charts.

    Handles:
    - Chart PNG uploads
    - Automatic bucket creation with lifecycle policy
    - Public URL generation
    - 30-day automatic expiration
    """

    # Default configuration
    DEFAULT_ENDPOINT = 'minio:9000'
    DEFAULT_BUCKET = 'claude-charts'
    DEFAULT_RETENTION_DAYS = 30

    def __init__(
        self,
        endpoint: str = None,
        access_key: str = None,
        secret_key: str = None,
        bucket_name: str = None,
        secure: bool = False,
        public_url: str = None
    ):
        """
        Initialize MinIO client.

        Args:
            endpoint: MinIO server endpoint (host:port)
            access_key: MinIO access key
            secret_key: MinIO secret key
            bucket_name: Bucket name for charts
            secure: Use HTTPS
            public_url: Public URL for generating accessible links
        """
        self.endpoint = endpoint or os.getenv('MINIO_ENDPOINT', self.DEFAULT_ENDPOINT)
        self.access_key = access_key or os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
        self.secret_key = secret_key or os.getenv('MINIO_SECRET_KEY', 'minioadmin123')
        self.bucket_name = bucket_name or os.getenv('MINIO_BUCKET_NAME', self.DEFAULT_BUCKET)
        self.secure = secure or os.getenv('MINIO_SECURE', 'false').lower() == 'true'
        self.public_url = public_url or os.getenv('MINIO_PUBLIC_URL', f"http://{self.endpoint}")
        self.retention_days = int(os.getenv('MINIO_CHART_RETENTION_DAYS', self.DEFAULT_RETENTION_DAYS))

        self.client = None
        self._initialized = False
        self._bucket_exists = False

        if not MINIO_SDK_AVAILABLE:
            logger.error("MinIO SDK not available - chart storage disabled")
            return

        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the MinIO client connection."""
        try:
            self.client = Minio(
                self.endpoint,
                access_key=self.access_key,
                secret_key=self.secret_key,
                secure=self.secure
            )
            self._initialized = True
            logger.info(f"MinIO client initialized - endpoint: {self.endpoint}, bucket: {self.bucket_name}")
        except Exception as e:
            logger.error(f"Failed to initialize MinIO client: {e}")
            self._initialized = False

    def _ensure_bucket_exists(self) -> bool:
        """
        Ensure the bucket exists with proper lifecycle policy.

        Returns:
            True if bucket is ready, False otherwise
        """
        if self._bucket_exists:
            return True

        if not self._initialized or not self.client:
            return False

        try:
            # Check if bucket exists
            if not self.client.bucket_exists(self.bucket_name):
                # Create the bucket
                self.client.make_bucket(self.bucket_name)
                logger.info(f"Created MinIO bucket: {self.bucket_name}")

                # Set lifecycle policy for 30-day expiration
                self._set_lifecycle_policy()

                # Set bucket policy for public read access
                self._set_public_read_policy()
            else:
                logger.debug(f"MinIO bucket already exists: {self.bucket_name}")

            self._bucket_exists = True
            return True

        except S3Error as e:
            logger.error(f"MinIO bucket error: {e}")
            return False
        except Exception as e:
            logger.error(f"Error ensuring bucket exists: {e}")
            return False

    def _set_lifecycle_policy(self) -> None:
        """Set lifecycle policy for automatic chart expiration."""
        try:
            # Create lifecycle rule for expiration
            config = LifecycleConfig(
                [
                    Rule(
                        rule_id="expire-charts",
                        status="Enabled",
                        expiration=Expiration(days=self.retention_days)
                    )
                ]
            )
            self.client.set_bucket_lifecycle(self.bucket_name, config)
            logger.info(f"Set {self.retention_days}-day lifecycle policy on bucket {self.bucket_name}")
        except Exception as e:
            logger.warning(f"Failed to set lifecycle policy: {e}")

    def _set_public_read_policy(self) -> None:
        """Set bucket policy to allow public read access for charts."""
        try:
            policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {"AWS": "*"},
                        "Action": ["s3:GetObject"],
                        "Resource": [f"arn:aws:s3:::{self.bucket_name}/*"]
                    }
                ]
            }
            self.client.set_bucket_policy(self.bucket_name, json.dumps(policy))
            logger.info(f"Set public read policy on bucket {self.bucket_name}")
        except Exception as e:
            logger.warning(f"Failed to set public read policy: {e}")

    @property
    def is_available(self) -> bool:
        """Check if MinIO client is available and initialized."""
        return self._initialized and self.client is not None

    def upload_chart(
        self,
        image_bytes: bytes,
        object_name: str,
        content_type: str = "image/png"
    ) -> Optional[str]:
        """
        Upload a chart image to MinIO.

        Args:
            image_bytes: PNG image data as bytes
            object_name: Object name (filename) in the bucket
            content_type: MIME type of the content

        Returns:
            Public URL of the uploaded chart, or None on failure
        """
        if not self.is_available:
            logger.warning("MinIO client not available - cannot upload chart")
            return None

        if not self._ensure_bucket_exists():
            logger.warning("MinIO bucket not available - cannot upload chart")
            return None

        try:
            # Create BytesIO object from bytes
            data = BytesIO(image_bytes)
            data_length = len(image_bytes)

            # Upload the object
            self.client.put_object(
                self.bucket_name,
                object_name,
                data,
                data_length,
                content_type=content_type
            )

            # Generate public URL
            url = self._get_public_url(object_name)
            logger.info(f"Uploaded chart to MinIO: {object_name} ({data_length} bytes)")

            return url

        except S3Error as e:
            logger.error(f"MinIO upload error: {e}")
            return None
        except Exception as e:
            logger.error(f"Error uploading chart to MinIO: {e}")
            return None

    def _get_public_url(self, object_name: str) -> str:
        """
        Generate public URL for an object.

        Args:
            object_name: Name of the object in the bucket

        Returns:
            Public URL for the object
        """
        # Use the public URL if configured, otherwise use endpoint
        base_url = self.public_url.rstrip('/')
        return f"{base_url}/{self.bucket_name}/{object_name}"

    def get_presigned_url(self, object_name: str, expires: int = 3600) -> Optional[str]:
        """
        Generate a presigned URL for temporary access.

        Args:
            object_name: Name of the object
            expires: URL expiration time in seconds

        Returns:
            Presigned URL or None on failure
        """
        if not self.is_available:
            return None

        try:
            url = self.client.presigned_get_object(
                self.bucket_name,
                object_name,
                expires=timedelta(seconds=expires)
            )
            return url
        except Exception as e:
            logger.error(f"Error generating presigned URL: {e}")
            return None

    def delete_chart(self, object_name: str) -> bool:
        """
        Delete a chart from MinIO.

        Args:
            object_name: Name of the object to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        if not self.is_available:
            return False

        try:
            self.client.remove_object(self.bucket_name, object_name)
            logger.info(f"Deleted chart from MinIO: {object_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting chart from MinIO: {e}")
            return False

    def object_exists(self, object_name: str) -> bool:
        """
        Check if an object exists in the bucket.

        Args:
            object_name: Name of the object

        Returns:
            True if exists, False otherwise
        """
        if not self.is_available:
            return False

        try:
            self.client.stat_object(self.bucket_name, object_name)
            return True
        except:
            return False

    def get_health_status(self) -> dict:
        """
        Get MinIO client health status.

        Returns:
            Dict with status information
        """
        if not MINIO_SDK_AVAILABLE:
            return {
                'status': 'unavailable',
                'reason': 'MinIO SDK not installed',
                'recommendation': 'Run: pip install minio'
            }

        if not self._initialized:
            return {
                'status': 'error',
                'reason': 'Client not initialized',
                'recommendation': 'Check MinIO endpoint and credentials'
            }

        try:
            # Try to list buckets as a health check
            buckets = list(self.client.list_buckets())
            bucket_exists = self.client.bucket_exists(self.bucket_name)

            return {
                'status': 'healthy',
                'endpoint': self.endpoint,
                'bucket': self.bucket_name,
                'bucket_exists': bucket_exists,
                'total_buckets': len(buckets),
                'retention_days': self.retention_days
            }
        except Exception as e:
            return {
                'status': 'error',
                'reason': str(e),
                'recommendation': 'Check MinIO server connectivity'
            }


# Singleton instance for reuse
_minio_client_instance: Optional[MinIOChartClient] = None


def get_minio_client() -> MinIOChartClient:
    """
    Get or create MinIO client singleton.

    Returns:
        MinIOChartClient instance
    """
    global _minio_client_instance

    if _minio_client_instance is None:
        _minio_client_instance = MinIOChartClient()

    return _minio_client_instance


def upload_vision_chart(
    image_bytes: bytes,
    alert_id: int,
    epic: str,
    timestamp: str
) -> Optional[str]:
    """
    Convenience function to upload a vision analysis chart.

    Args:
        image_bytes: PNG image data
        alert_id: Alert ID for filename
        epic: Trading pair epic
        timestamp: Timestamp string for filename

    Returns:
        Public URL of the uploaded chart, or None on failure
    """
    client = get_minio_client()

    # Clean epic for filename
    epic_clean = epic.replace('.', '_').replace(':', '_')

    # Create object name
    object_name = f"{alert_id}_{epic_clean}_{timestamp}_chart.png"

    return client.upload_chart(image_bytes, object_name)
