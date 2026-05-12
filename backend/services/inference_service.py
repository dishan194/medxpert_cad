"""
backend/services/inference_service.py
Orchestrates the full inference pipeline:
  Upload → Preprocess → Classify → Grad-CAM → Segment → Report
Manages model lifecycle and caching.
"""

import time
import uuid
import base64
import io
import logging
import numpy as np
import torch
from typing import Dict, Optional, Any
from PIL import Image

from backend.config.settings import get_settings
from backend.utils.dicom_utils import DICOMProcessor
from backend.utils.audit_logger import get_audit_logger
from backend.models.xray_classifier import XRayClassifier
from backend.models.mri_classifier import MRIClassifier
from backend.models.grad_cam import build_gradcam
from backend.models.segmentation import MaskRCNNSegmenter
from backend.models.report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class InferenceService:
    """
    Singleton service that manages all ML models and runs end-to-end inference.
    """

    def __init__(self):
        self.settings = get_settings()
        self.audit    = get_audit_logger()
        self.dicom    = DICOMProcessor(
            target_size=(self.settings.image_size, self.settings.image_size)
        )

        logger.info("Loading models...")
        self._load_models()
        logger.info("All models ready.")

    def _load_models(self):
        s = self.settings
        device = s.device

        self.xray_classifier = XRayClassifier(
            model_path=s.xray_model_path if self._file_exists(s.xray_model_path) else None,
            model_type="resnet50",
            device=device,
            threshold=s.classification_threshold,
        )

        self.mri_classifier = MRIClassifier(
            model_path=s.mri_model_path if self._file_exists(s.mri_model_path) else None,
            device=device,
        )

        self.segmenter = MaskRCNNSegmenter(
            model_path=s.segmentation_model_path if self._file_exists(s.segmentation_model_path) else None,
            device=device,
        )

        self.report_generator = ReportGenerator(
            model_path=s.report_model_path if self._file_exists(s.report_model_path) else None,
            device=device,
        )

    @staticmethod
    def _file_exists(path: str) -> bool:
        import os
        return os.path.isfile(path)

    # ──────────────────────────────────────────────────────────
    # Main Inference Entry Points
    # ──────────────────────────────────────────────────────────

    def analyze_xray(
        self,
        file_bytes: bytes,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        run_segmentation: bool = True,
    ) -> Dict[str, Any]:
        """
        Full chest X-ray analysis pipeline.
        Returns classification, Grad-CAM heatmap, segmentation, and report.
        """
        session_id = session_id or str(uuid.uuid4())
        start_time = time.time()

        # Audit upload
        self.audit.log_upload(session_id, ip_address or "unknown", "xray", len(file_bytes))

        try:
            # 1. Preprocess
            original_np = self._bytes_to_numpy(file_bytes)
            image_tensor = torch.from_numpy(
                self.dicom.preprocess_for_model(original_np)
            ).float().unsqueeze(0)

            # 2. Classification
            classification = self.xray_classifier.predict(image_tensor)

            # 3. Grad-CAM heatmap
            heatmap_b64 = self._generate_gradcam_xray(
                image_tensor, original_np, classification
            )

            # 4. Segmentation (if abnormalities found)
            segmentation = {}
            seg_vis_b64  = None
            if run_segmentation and not classification["is_normal"]:
                segmentation, seg_vis_b64 = self._run_segmentation(original_np)

            # 5. Report Generation
            report = self.report_generator.generate_structured_report(
                image_tensor,
                classification_result=classification,
                segmentation_result=segmentation,
                modality="Chest X-Ray",
            )

            elapsed_ms = (time.time() - start_time) * 1000
            self.audit.log_inference(
                session_id, "xray", "resnet50-v1.0",
                elapsed_ms, len(classification["findings"])
            )

            return {
                "session_id": session_id,
                "modality": "xray",
                "classification": classification,
                "heatmap_base64": heatmap_b64,
                "segmentation": segmentation,
                "segmentation_vis_base64": seg_vis_b64,
                "report": report,
                "inference_time_ms": round(elapsed_ms, 2),
                "original_image_base64": self._numpy_to_base64(original_np),
            }

        except Exception as e:
            self.audit.log_error(session_id, type(e).__name__, str(e))
            raise

    def analyze_mri(
        self,
        file_bytes: bytes,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        run_segmentation: bool = True,
    ) -> Dict[str, Any]:
        """Full MRI brain analysis pipeline."""
        session_id = session_id or str(uuid.uuid4())
        start_time = time.time()

        self.audit.log_upload(session_id, ip_address or "unknown", "mri", len(file_bytes))

        try:
            original_np  = self._bytes_to_numpy(file_bytes)
            image_tensor = torch.from_numpy(
                self.dicom.preprocess_for_model(original_np)
            ).float().unsqueeze(0)

            classification = self.mri_classifier.predict(image_tensor)

            heatmap_b64 = self._generate_gradcam_mri(
                image_tensor, original_np, classification
            )

            segmentation = {}
            seg_vis_b64  = None
            if run_segmentation and not classification["is_normal"]:
                segmentation, seg_vis_b64 = self._run_segmentation(original_np)

            report = self.report_generator.generate_structured_report(
                image_tensor,
                classification_result={
                    "is_normal": classification["is_normal"],
                    "findings": [] if classification["is_normal"] else [{
                        "pathology": classification["predicted_class"],
                        "probability": classification["confidence"],
                        "severity": classification["severity"],
                    }],
                    "overall_confidence": classification["confidence"],
                },
                segmentation_result=segmentation,
                modality="MRI Brain",
            )

            elapsed_ms = (time.time() - start_time) * 1000
            self.audit.log_inference(
                session_id, "mri", "resnet50-mri-v1.0",
                elapsed_ms, 0 if classification["is_normal"] else 1
            )

            return {
                "session_id": session_id,
                "modality": "mri",
                "classification": classification,
                "heatmap_base64": heatmap_b64,
                "segmentation": segmentation,
                "segmentation_vis_base64": seg_vis_b64,
                "report": report,
                "inference_time_ms": round(elapsed_ms, 2),
                "original_image_base64": self._numpy_to_base64(original_np),
            }

        except Exception as e:
            self.audit.log_error(session_id, type(e).__name__, str(e))
            raise

    # ──────────────────────────────────────────────────────────
    # Internal Helpers
    # ──────────────────────────────────────────────────────────

    def _bytes_to_numpy(self, file_bytes: bytes) -> np.ndarray:
        """Load image from bytes → (H, W) or (H, W, 3) uint8 array."""
        try:
            # Try PIL (JPEG, PNG)
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            return np.array(img)
        except Exception:
            pass
        # Try DICOM
        from backend.utils.dicom_utils import DICOMProcessor
        proc = DICOMProcessor()
        pixel_array, _ = proc.load_dicom(file_bytes)
        normalized = proc.normalize_xray(pixel_array)
        return normalized

    def _generate_gradcam_xray(
        self,
        image_tensor: torch.Tensor,
        original_np: np.ndarray,
        classification: Dict,
    ) -> Optional[str]:
        try:
            gradcam = build_gradcam(self.xray_classifier.model, "resnet50")
            image_tensor.requires_grad_(True)

            if classification["is_normal"]:
                return None

            # Use top predicted class for heatmap
            top_finding = classification["findings"][0]["pathology"]
            class_idx   = self.xray_classifier.PATHOLOGY_LABELS.index(top_finding)

            h, w = original_np.shape[:2] if original_np.ndim >= 2 else (224, 224)
            heatmap = gradcam.generate(image_tensor, class_idx, (h, w))
            overlaid = gradcam.overlay_on_image(original_np, heatmap)
            return gradcam.to_base64(overlaid)
        except Exception as e:
            logger.warning(f"Grad-CAM failed: {e}")
            return None

    def _generate_gradcam_mri(
        self,
        image_tensor: torch.Tensor,
        original_np: np.ndarray,
        classification: Dict,
    ) -> Optional[str]:
        try:
            gradcam = build_gradcam(self.mri_classifier.model, "resnet50")
            image_tensor.requires_grad_(True)

            if classification["is_normal"]:
                return None

            class_idx = self.mri_classifier.TUMOR_LABELS.index(
                classification["predicted_class"]
            )

            h, w = original_np.shape[:2] if original_np.ndim >= 2 else (224, 224)
            heatmap = gradcam.generate(image_tensor, class_idx, (h, w))
            overlaid = gradcam.overlay_on_image(original_np, heatmap)
            return gradcam.to_base64(overlaid)
        except Exception as e:
            logger.warning(f"MRI Grad-CAM failed: {e}")
            return None

    def _run_segmentation(self, image_np: np.ndarray):
        try:
            result = self.segmenter.predict(image_np)
            vis    = self.segmenter.visualize(image_np, result)
            vis_b64 = self._numpy_to_base64(vis)
            seg_dict = self.segmenter.result_to_dict(result)
            return seg_dict, vis_b64
        except Exception as e:
            logger.warning(f"Segmentation failed: {e}")
            return {}, None

    @staticmethod
    def _numpy_to_base64(image: np.ndarray) -> str:
        if image is None:
            return ""
        if image.dtype != np.uint8:
            image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        pil_img = Image.fromarray(image)
        buffer  = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ─────────────────────────────────────────────────────────────
# Singleton accessor
# ─────────────────────────────────────────────────────────────

_service_instance: Optional[InferenceService] = None


def get_inference_service() -> InferenceService:
    global _service_instance
    if _service_instance is None:
        _service_instance = InferenceService()
    return _service_instance