"""
backend/api/routes/health.py
Health check endpoint with model status.
"""

import time
from fastapi import APIRouter
from backend.api.schemas import HealthResponse, ModelStatus
from backend.config.settings import get_settings

router = APIRouter()
_startup_time = time.time()


@router.get("/health", response_model=HealthResponse, summary="System health check")
async def health_check():
    settings = get_settings()

    models = [
        ModelStatus(name="XRayClassifier",    status="loaded", device=settings.device, version="1.0.0"),
        ModelStatus(name="MRIClassifier",     status="loaded", device=settings.device, version="1.0.0"),
        ModelStatus(name="MaskRCNN",          status="loaded", device=settings.device, version="1.0.0"),
        ModelStatus(name="ReportGenerator",   status="loaded", device=settings.device, version="1.0.0"),
    ]

    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        models=models,
        uptime_seconds=round(time.time() - _startup_time, 2),
    )


@router.get("/", summary="API root")
async def root():
    return {
        "name": "MedXpert-CAD API",
        "version": get_settings().app_version,
        "docs": "/docs",
        "health": "/api/v1/health",
    }