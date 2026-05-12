"""
backend/utils/dicom_utils.py
DICOM loading, preprocessing, and normalization utilities.
"""

import io
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional
import logging

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False

try:
    import SimpleITK as sitk
    SITK_AVAILABLE = True
except ImportError:
    SITK_AVAILABLE = False

from PIL import Image
import cv2

logger = logging.getLogger(__name__)


class DICOMProcessor:
    """
    Handles DICOM file loading, windowing, and normalization.
    Supports X-ray and MRI modalities.
    """

    # Standard windowing presets (center, width)
    WINDOW_PRESETS = {
        "chest":     (-600, 1500),   # Lung window
        "mediastinum": (50, 350),    # Mediastinal window
        "bone":      (300, 1500),    # Bone window
        "brain":     (40, 80),       # Brain window
        "subdural":  (75, 215),
        "stroke":    (32, 8),
    }

    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size

    def load_dicom(self, path: Union[str, Path, bytes]) -> Optional[np.ndarray]:
        """Load a DICOM file and return pixel array."""
        if not PYDICOM_AVAILABLE:
            raise ImportError("pydicom is required. Install with: pip install pydicom")

        try:
            if isinstance(path, bytes):
                ds = pydicom.dcmread(io.BytesIO(path))
            else:
                ds = pydicom.dcmread(str(path))

            pixel_array = ds.pixel_array.astype(np.float32)

            # Apply VOI LUT if present (windowing from DICOM header)
            if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
                pixel_array = apply_voi_lut(pixel_array, ds)

            # Handle PhotometricInterpretation
            if hasattr(ds, 'PhotometricInterpretation'):
                if ds.PhotometricInterpretation == "MONOCHROME1":
                    pixel_array = np.max(pixel_array) - pixel_array

            # Rescale slope/intercept
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                pixel_array = pixel_array * float(ds.RescaleSlope) + float(ds.RescaleIntercept)

            return pixel_array, ds

        except Exception as e:
            logger.error(f"Failed to load DICOM: {e}")
            raise

    def apply_windowing(
        self,
        pixel_array: np.ndarray,
        window_center: float,
        window_width: float
    ) -> np.ndarray:
        """Apply CT windowing to pixel array."""
        lower = window_center - window_width / 2
        upper = window_center + window_width / 2
        windowed = np.clip(pixel_array, lower, upper)
        # Normalize to [0, 255]
        windowed = ((windowed - lower) / (upper - lower) * 255).astype(np.uint8)
        return windowed

    def normalize_xray(self, pixel_array: np.ndarray) -> np.ndarray:
        """Normalize chest X-ray pixel values to [0, 255]."""
        p_min, p_max = np.percentile(pixel_array, [1, 99])
        clipped = np.clip(pixel_array, p_min, p_max)
        normalized = ((clipped - p_min) / (p_max - p_min + 1e-8) * 255).astype(np.uint8)
        return normalized

    def normalize_mri(self, pixel_array: np.ndarray, preset: str = "brain") -> np.ndarray:
        """Normalize MRI with windowing preset."""
        center, width = self.WINDOW_PRESETS.get(preset, (40, 80))
        return self.apply_windowing(pixel_array, center, width)

    def to_rgb(self, pixel_array: np.ndarray) -> np.ndarray:
        """Convert grayscale to 3-channel RGB."""
        if pixel_array.ndim == 2:
            return np.stack([pixel_array] * 3, axis=-1)
        return pixel_array

    def resize(self, image: np.ndarray) -> np.ndarray:
        """Resize image to target size using INTER_LANCZOS4."""
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_LANCZOS4)

    def preprocess_for_model(
        self,
        image: np.ndarray,
        augment: bool = False
    ) -> np.ndarray:
        """
        Full preprocessing pipeline:
        grayscale → normalize → RGB → resize → [0,1] float32
        """
        # Ensure uint8
        if image.dtype != np.uint8:
            image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)

        rgb = self.to_rgb(image)
        resized = self.resize(rgb)

        if augment:
            resized = self._augment(resized)

        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std  = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std

        # HWC → CHW
        return normalized.transpose(2, 0, 1)

    def _augment(self, image: np.ndarray) -> np.ndarray:
        """Light augmentation for training robustness."""
        try:
            import albumentations as A
            transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(var_limit=(0, 10), p=0.2),
                A.Rotate(limit=10, p=0.3),
            ])
            return transform(image=image)["image"]
        except ImportError:
            return image

    def load_from_upload(self, file_bytes: bytes, modality: str = "xray") -> np.ndarray:
        """
        Load from uploaded bytes (DICOM or standard image).
        Returns preprocessed numpy array ready for model.
        """
        # Try DICOM first
        try:
            pixel_array, _ = self.load_dicom(file_bytes)
            if modality == "xray":
                pixel_array = self.normalize_xray(pixel_array)
            else:
                pixel_array = self.normalize_mri(pixel_array)
            return self.preprocess_for_model(pixel_array)
        except Exception:
            pass

        # Fallback: standard image (PNG/JPEG)
        try:
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
            image_np = np.array(image)
            return self.preprocess_for_model(image_np)
        except Exception as e:
            raise ValueError(f"Cannot load image: {e}")

    def extract_metadata(self, dicom_bytes: bytes) -> dict:
        """Extract anonymized metadata from DICOM header."""
        if not PYDICOM_AVAILABLE:
            return {}
        try:
            ds = pydicom.dcmread(io.BytesIO(dicom_bytes))
            return {
                "modality": getattr(ds, "Modality", "Unknown"),
                "study_description": getattr(ds, "StudyDescription", ""),
                "series_description": getattr(ds, "SeriesDescription", ""),
                "rows": getattr(ds, "Rows", 0),
                "columns": getattr(ds, "Columns", 0),
                "pixel_spacing": list(getattr(ds, "PixelSpacing", [1.0, 1.0])),
                "bits_allocated": getattr(ds, "BitsAllocated", 16),
                # NOTE: Patient identifiers are NOT included (HIPAA)
            }
        except Exception:
            return {}