"""
backend/config/settings.py
Central configuration loaded from environment variables.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    # Application
    app_name: str = "MedXpert-CAD"
    app_version: str = "1.0.0"
    debug: bool = False
    secret_key: str = "change-me-in-production"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4

    # Database
    database_url: str = "postgresql://user:password@localhost:5432/medxpert"
    redis_url: str = "redis://localhost:6379/0"

    # AWS
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"
    s3_bucket_name: str = "medxpert-dicom-storage"
    aws_endpoint_url: Optional[str] = None

    # Model Paths
    xray_model_path: str = "data/models/xray_classifier.pth"
    mri_model_path: str = "data/models/mri_classifier.pth"
    segmentation_model_path: str = "data/models/mask_rcnn.pth"
    report_model_path: str = "data/models/report_generator.pth"

    # Model Config
    device: str = "cuda"
    batch_size: int = 16
    image_size: int = 224
    num_classes_xray: int = 14
    num_classes_mri: int = 5

    # HIPAA / Compliance
    audit_log_path: str = "logs/audit.log"
    enable_audit_logging: bool = True
    data_retention_days: int = 365

    # Thresholds
    classification_threshold: float = 0.5
    confidence_min_display: float = 0.1

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()


# NIH ChestX-ray14 class labels
XRAY_PATHOLOGY_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
    "Mass", "Nodule", "Pneumonia", "Pneumothorax",
    "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]

# MRI pathology labels
MRI_PATHOLOGY_LABELS = [
    "Normal", "Glioma", "Meningioma", "Pituitary_Tumor", "Metastasis"
]

# Clinical severity mapping
SEVERITY_MAP = {
    "Pneumothorax": "critical",
    "Pneumonia": "high",
    "Cardiomegaly": "high",
    "Effusion": "moderate",
    "Atelectasis": "moderate",
    "Consolidation": "high",
    "Edema": "high",
    "Glioma": "critical",
    "Meningioma": "high",
    "Pituitary_Tumor": "moderate",
    "Metastasis": "critical",
}