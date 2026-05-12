"""
backend/models/mri_classifier.py
MRI brain tumor multi-class classifier.
Uses VGG16 / ResNet50 with transfer learning.
Classes: Normal, Glioma, Meningioma, Pituitary Tumor, Metastasis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class VGG16Classifier(nn.Module):
    """
    VGG16-based MRI tumor classifier.
    Fine-tuned on top of ImageNet weights.
    """

    def __init__(self, num_classes: int = 5, pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        self.num_classes = num_classes

        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.vgg16(weights=weights)

        # Feature extractor (conv layers)
        self.features = backbone.features
        self.avgpool  = backbone.avgpool  # (7, 7) adaptive

        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.6),
            nn.Linear(256, num_classes),
        )

        # Freeze early layers (fine-tune only last conv block + head)
        for i, layer in enumerate(self.features):
            if i < 24:  # Freeze first 4 blocks
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.features(x)       # (B, 512, 7, 7)
        pooled   = self.avgpool(features) # (B, 512, 7, 7)
        flat     = pooled.flatten(1)      # (B, 512*7*7)
        logits   = self.classifier(flat)  # (B, num_classes)
        return logits, features


class ResNet50MRIClassifier(nn.Module):
    """
    ResNet50-based MRI tumor classifier with attention mechanism.
    """

    def __init__(self, num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = models.resnet50(weights=weights)

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Channel attention (SE block)
        self.se_block = SqueezeExcitation(2048, reduction=16)

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)  # (B, 2048, 7, 7)
        features = self.se_block(features)     # attention-weighted
        pooled   = self.avgpool(features)
        flat     = pooled.flatten(1)
        logits   = self.classifier(flat)
        return logits, features


class SqueezeExcitation(nn.Module):
    """Channel-wise attention block (SE-Net)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(x).view(x.shape[0], x.shape[1], 1, 1)
        return x * scale


class MRIClassifier:
    """
    High-level wrapper for MRI tumor classification inference.
    """

    TUMOR_LABELS = ["Normal", "Glioma", "Meningioma", "Pituitary_Tumor", "Metastasis"]

    TUMOR_DESCRIPTIONS = {
        "Normal":          "No significant abnormality detected.",
        "Glioma":          "Glioma suspected — malignant brain tumor arising from glial cells.",
        "Meningioma":      "Meningioma suspected — typically benign tumor of the meninges.",
        "Pituitary_Tumor": "Pituitary adenoma suspected — usually benign, may cause hormonal effects.",
        "Metastasis":      "Brain metastasis suspected — secondary tumor from distant primary site.",
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_type: str = "resnet50",
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model_version = "1.0.0"

        if model_type == "vgg16":
            self.model = VGG16Classifier(num_classes=len(self.TUMOR_LABELS))
        else:
            self.model = ResNet50MRIClassifier(num_classes=len(self.TUMOR_LABELS))

        self.model = self.model.to(self.device)

        if model_path:
            self._load_weights(model_path)

        self.model.eval()
        logger.info(f"MRIClassifier loaded ({model_type}) on {self.device}")

    def _load_weights(self, path: str):
        try:
            state_dict = torch.load(path, map_location=self.device)
            if any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"MRI model weights loaded from {path}")
        except FileNotFoundError:
            logger.warning(f"Weight file not found: {path}")

    @torch.no_grad()
    def predict(self, image_tensor: torch.Tensor) -> Dict:
        """Run MRI tumor classification inference."""
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(self.device)
        logits, features = self.model(image_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()

        predicted_idx = int(probs[0].argmax())
        predicted_class = self.TUMOR_LABELS[predicted_idx]
        confidence = float(probs[0, predicted_idx])

        all_probs = {
            label: round(float(probs[0, i]), 4)
            for i, label in enumerate(self.TUMOR_LABELS)
        }

        from backend.config.settings import SEVERITY_MAP
        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "all_probabilities": all_probs,
            "description": self.TUMOR_DESCRIPTIONS[predicted_class],
            "severity": SEVERITY_MAP.get(predicted_class, "low"),
            "is_normal": predicted_class == "Normal",
            "model_version": self.model_version,
        }

    def get_feature_maps(self, image_tensor: torch.Tensor) -> torch.Tensor:
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            _, features = self.model(image_tensor)
        return features