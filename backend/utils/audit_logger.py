"""
backend/utils/audit_logger.py
HIPAA-compliant audit logging for all PHI access and model inference events.
"""

import json
import uuid
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any
from enum import Enum

from loguru import logger as loguru_logger


class AuditEventType(str, Enum):
    IMAGE_UPLOAD      = "IMAGE_UPLOAD"
    INFERENCE_REQUEST = "INFERENCE_REQUEST"
    INFERENCE_RESULT  = "INFERENCE_RESULT"
    REPORT_GENERATED  = "REPORT_GENERATED"
    REPORT_ACCESSED   = "REPORT_ACCESSED"
    MODEL_ERROR       = "MODEL_ERROR"
    AUTH_SUCCESS      = "AUTH_SUCCESS"
    AUTH_FAILURE      = "AUTH_FAILURE"
    DATA_DELETION     = "DATA_DELETION"
    SYSTEM_STARTUP    = "SYSTEM_STARTUP"


class AuditLogger:
    """
    HIPAA-compliant audit logger.
    Logs: who accessed what, when, from where, and with what result.
    Never logs raw PHI (patient names, DOBs, MRNs).
    """

    def __init__(self, log_path: str = "logs/audit.log", enabled: bool = True):
        self.enabled = enabled
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        if enabled:
            loguru_logger.add(
                str(self.log_path),
                format="{message}",
                rotation="100 MB",
                retention="365 days",
                compression="gz",
                serialize=False,
                filter=lambda record: record["extra"].get("audit", False),
            )

    def _build_record(
        self,
        event_type: AuditEventType,
        session_id: Optional[str],
        user_id: Optional[str],
        ip_address: Optional[str],
        resource_id: Optional[str],
        details: Optional[Dict[str, Any]] = None,
        success: bool = True,
    ) -> dict:
        return {
            "audit_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type.value,
            "session_id": session_id or "anonymous",
            "user_id_hash": hashlib.sha256((user_id or "").encode()).hexdigest()[:16] if user_id else None,
            "ip_address": ip_address,
            "resource_id": resource_id,
            "success": success,
            "details": details or {},
        }

    def log(
        self,
        event_type: AuditEventType,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        success: bool = True,
    ):
        if not self.enabled:
            return

        record = self._build_record(
            event_type, session_id, user_id,
            ip_address, resource_id, details, success
        )
        loguru_logger.bind(audit=True).info(json.dumps(record))

    def log_upload(self, session_id: str, ip: str, modality: str, file_size_bytes: int):
        self.log(
            AuditEventType.IMAGE_UPLOAD,
            session_id=session_id,
            ip_address=ip,
            details={"modality": modality, "file_size_bytes": file_size_bytes},
        )

    def log_inference(self, session_id: str, modality: str, model_version: str,
                      inference_ms: float, findings_count: int, success: bool = True):
        self.log(
            AuditEventType.INFERENCE_RESULT,
            session_id=session_id,
            details={
                "modality": modality,
                "model_version": model_version,
                "inference_time_ms": round(inference_ms, 2),
                "findings_count": findings_count,
            },
            success=success,
        )

    def log_report(self, session_id: str, report_id: str, bleu_score: Optional[float] = None):
        self.log(
            AuditEventType.REPORT_GENERATED,
            session_id=session_id,
            resource_id=report_id,
            details={"bleu_score": bleu_score},
        )

    def log_error(self, session_id: str, error_type: str, message: str):
        self.log(
            AuditEventType.MODEL_ERROR,
            session_id=session_id,
            details={"error_type": error_type, "message": message[:200]},
            success=False,
        )


# Singleton instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        from backend.config.settings import get_settings
        s = get_settings()
        _audit_logger = AuditLogger(
            log_path=s.audit_log_path,
            enabled=s.enable_audit_logging,
        )
    return _audit_logger