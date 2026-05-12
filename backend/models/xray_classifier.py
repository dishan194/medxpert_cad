"""
backend/models/xray_classifier.py
Multi-label chest X-ray classifier.
Week 1-3: ResNet50 → Week 3+: Vision Transformer (ViT)
Supports 14 NIH ChestX-ray14 pathologies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ResNet50Classifier(nn.Module):
    """
    ResNet50-based multi-label chest X-ray classifier.
    Uses ImageNet pretrained weights + custom classification head.
    Week 1-2: Binary (normal vs abnormal)
    Week 3+:  Multi-label (14 pathologies)
    """

    def __init__(self, num_classes: int = 14, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        self.num_classes = num_classes

        # Load pretrained ResNet50
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Remove final FC layer, keep feature extractor
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_classes),
        )

        # Feature dimension for Grad-CAM
        self.feature_dim = 2048

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: (B, num_classes) — raw scores
            features: (B, 2048, H, W) — spatial features for Grad-CAM
        """
        features = self.feature_extractor(x)   # (B, 2048, 7, 7)
        pooled = self.avgpool(features)          # (B, 2048, 1, 1)
        flat = pooled.flatten(1)                 # (B, 2048)
        logits = self.classifier(flat)           # (B, num_classes)
        return logits, features

    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities for multi-label classification."""
        logits, _ = self.forward(x)
        return torch.sigmoid(logits)


class ViTClassifier(nn.Module):
    """
    Vision Transformer (ViT-B/16) for chest X-ray classification.
    Week 3+ upgrade for better feature extraction.
    Uses timm library for pretrained ViT weights.
    """

    def __init__(self, num_classes: int = 14, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.num_classes = num_classes

        try:
            import timm
            self.backbone = timm.create_model(
                'vit_base_patch16_224',
                pretrained=pretrained,
                num_classes=0,  # Remove head, we add our own
                drop_rate=dropout,
            )
            embed_dim = self.backbone.embed_dim  # 768 for ViT-B
        except ImportError:
            raise ImportError("timm is required for ViT. Install: pip install timm")

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Returns:
            logits: (B, num_classes)
            None: ViT doesn't expose spatial feature maps the same way
        """
        cls_token = self.backbone(x)   # (B, embed_dim)
        logits = self.classifier(cls_token)
        return logits, None

    def get_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(x)
        return torch.sigmoid(logits)


class XRayClassifier:
    """
    High-level wrapper for X-ray classification inference.
    Handles model loading, device management, and prediction formatting.
    """

    PATHOLOGY_LABELS = [
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration",
        "Mass", "Nodule", "Pneumonia", "Pneumothorax",
        "Consolidation", "Edema", "Emphysema", "Fibrosis",
        "Pleural_Thickening", "Hernia"
    ]

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = "resnet50",   # "resnet50" or "vit"
        device: str = "cuda",
        threshold: float = 0.5,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.model_type = model_type
        self.model_version = "1.0.0"

        # Build model
        if model_type == "vit":
            self.model = ViTClassifier(num_classes=len(self.PATHOLOGY_LABELS))
        else:
            self.model = ResNet50Classifier(num_classes=len(self.PATHOLOGY_LABELS))

        self.model = self.model.to(self.device)

        # Load weights if provided
        if model_path:
            self._load_weights(model_path)

        self.model.eval()
        logger.info(f"XRayClassifier loaded ({model_type}) on {self.device}")

    def _load_weights(self, path: str):
        try:
            state_dict = torch.load(path, map_location=self.device)
            # Handle DataParallel wrapping
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded weights from {path}")
        except FileNotFoundError:
            logger.warning(f"Weight file not found: {path}. Using ImageNet pretrained weights.")
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")

    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor) -> Dict:
        """
        Run inference on a preprocessed image tensor.

        Args:
            image_tensor: (1, 3, 224, 224) or (B, 3, 224, 224)

        Returns:
            dict with findings, probabilities, severity
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)

        logits, features = self.model(image_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()

        findings = []
        all_scores = {}

        for i, label in enumerate(self.PATHOLOGY_LABELS):
            score = float(probs[0, i])
            all_scores[label] = round(score, 4)

            if score >= self.threshold:
                from backend.config.settings import SEVERITY_MAP
                findings.append({
                    "pathology": label,
                    "probability": round(score, 4),
                    "severity": SEVERITY_MAP.get(label, "low"),
                })

        # Sort by probability descending
        findings.sort(key=lambda x: x["probability"], reverse=True)

        is_normal = len(findings) == 0
        overall_confidence = float(max(probs[0])) if not is_normal else float(1 - max(probs[0]))

        return {
            "is_normal": is_normal,
            "findings": findings,
            "all_scores": all_scores,
            "overall_confidence": round(overall_confidence, 4),
            "model_type": self.model_type,
            "model_version": self.model_version,
        }

    def get_feature_maps(self, image_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """Return spatial feature maps for Grad-CAM (ResNet only)."""
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)

        if isinstance(self.model, ResNet50Classifier):
            with torch.no_grad():
                _, features = self.model(image_tensor)
            return features
        return None