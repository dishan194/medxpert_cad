"""
backend/api/schemas.py
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Any
from enum import Enum


class ModalityEnum(str, Enum):
    xray = "xray"
    mri  = "mri"


# ─────────────────────────────────────────────────────────────
# Classification Schemas
# ─────────────────────────────────────────────────────────────

class FindingItem(BaseModel):
    pathology:   str
    probability: float = Field(ge=0.0, le=1.0)
    severity:    str   = Field(default="low")


class ClassificationResult(BaseModel):
    is_normal:          bool
    findings:           List[FindingItem] = []
    all_scores:         Optional[Dict[str, float]] = None
    all_probabilities:  Optional[Dict[str, float]] = None
    overall_confidence: float
    predicted_class:    Optional[str] = None   # MRI only
    description:        Optional[str] = None   # MRI only
    model_type:         Optional[str] = None
    model_version:      str = "1.0.0"


# ─────────────────────────────────────────────────────────────
# Segmentation Schemas
# ─────────────────────────────────────────────────────────────

class SegmentationInstance(BaseModel):
    label:             str
    score:             float
    box:               List[float]  # [x1, y1, x2, y2]
    mask_area_pixels:  int


class SegmentationResult(BaseModel):
    num_instances: int = 0
    instances:     List[SegmentationInstance] = []


# ─────────────────────────────────────────────────────────────
# Report Schemas
# ─────────────────────────────────────────────────────────────

class ReportResult(BaseModel):
    report_text:    str
    findings_text:  str
    impression:     str
    confidence:     float
    model_version:  str = "1.0.0"


# ─────────────────────────────────────────────────────────────
# Analysis Response Schema
# ─────────────────────────────────────────────────────────────

class AnalysisResponse(BaseModel):
    session_id:                  str
    modality:                    ModalityEnum
    classification:              ClassificationResult
    heatmap_base64:              Optional[str] = None
    segmentation:                SegmentationResult = SegmentationResult()
    segmentation_vis_base64:     Optional[str] = None
    report:                      ReportResult
    original_image_base64:       Optional[str] = None
    inference_time_ms:           float
    warning:                     Optional[str] = None

    class Config:
        use_enum_values = True


# ─────────────────────────────────────────────────────────────
# Health Schemas
# ─────────────────────────────────────────────────────────────

class ModelStatus(BaseModel):
    name:    str
    status:  str  # "loaded" | "error" | "loading"
    device:  Optional[str] = None
    version: Optional[str] = None


class HealthResponse(BaseModel):
    status:        str  # "healthy" | "degraded" | "unhealthy"
    version:       str
    models:        List[ModelStatus]
    uptime_seconds: float


# ─────────────────────────────────────────────────────────────
# BLEU Evaluation Schema
# ─────────────────────────────────────────────────────────────

class BLEUEvalRequest(BaseModel):
    generated_report: str
    reference_reports: List[str] = Field(min_length=1)


class BLEUEvalResponse(BaseModel):
    bleu1: float
    bleu2: float
    bleu4: float
    meets_sota: bool  # True if bleu4 >= 0.415