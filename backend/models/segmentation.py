"""
backend/models/segmentation.py
Mask R-CNN for instance segmentation of abnormal regions.
Identifies and highlights specific pathological areas in images.
Week 4-5 Advanced Level feature.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SegmentationResult:
    boxes: np.ndarray         # (N, 4) xyxy format
    masks: np.ndarray         # (N, H, W) binary masks
    labels: List[str]
    scores: np.ndarray        # (N,) confidence scores
    num_instances: int


class MaskRCNNSegmenter:
    """
    Mask R-CNN for medical image segmentation.
    Uses torchvision's pretrained Mask R-CNN backbone.
    Fine-tuned for medical abnormality detection.
    """

    # Medical region labels (maps to COCO-like class indices for fine-tuning)
    MEDICAL_LABELS = {
        1:  "Nodule",
        2:  "Mass",
        3:  "Effusion_Region",
        4:  "Atelectasis_Region",
        5:  "Infiltration_Region",
        6:  "Tumor_Region",
        7:  "Normal_Region",
    }

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        num_classes: int = 8,   # background + 7 medical classes
        score_threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.score_threshold = score_threshold
        self.mask_threshold  = mask_threshold
        self.num_classes     = num_classes

        self.model = self._build_model(num_classes, pretrained=(model_path is None))
        self.model = self.model.to(self.device)

        if model_path:
            self._load_weights(model_path)

        self.model.eval()
        logger.info(f"MaskRCNN loaded on {self.device}")

    def _build_model(self, num_classes: int, pretrained: bool = True) -> nn.Module:
        """Build Mask R-CNN with custom classification head."""
        from torchvision.models.detection import (
            maskrcnn_resnet50_fpn,
            MaskRCNN_ResNet50_FPN_Weights,
        )
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None
        model = maskrcnn_resnet50_fpn(weights=weights)

        # Replace box predictor
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Replace mask predictor
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        model.roi_heads.mask_predictor = MaskRCNNPredictor(
            in_features_mask, hidden_layer, num_classes
        )

        return model

    def _load_weights(self, path: str):
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"Segmentation weights loaded from {path}")
        except FileNotFoundError:
            logger.warning(f"Weight file not found: {path}")

    def _preprocess(self, image: np.ndarray) -> List[torch.Tensor]:
        """Convert numpy image to Mask R-CNN input format."""
        # Ensure float32 in [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.max() > 1.0:
            image = image / image.max()

        # (H, W, 3) → (3, H, W)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()
        return [tensor.to(self.device)]

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> SegmentationResult:
        """
        Run Mask R-CNN on an image.

        Args:
            image: (H, W, 3) uint8 or float32 image

        Returns:
            SegmentationResult with boxes, masks, labels, scores
        """
        h, w = image.shape[:2]
        inputs = self._preprocess(image)

        predictions = self.model(inputs)[0]

        boxes  = predictions["boxes"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        masks  = predictions["masks"].cpu().numpy()  # (N, 1, H, W)

        # Filter by confidence threshold
        keep = scores >= self.score_threshold
        boxes  = boxes[keep]
        labels = labels[keep]
        scores = scores[keep]
        masks  = masks[keep]

        # Binarize masks
        binary_masks = (masks[:, 0] > self.mask_threshold).astype(np.uint8)  # (N, H, W)

        label_names = [self.MEDICAL_LABELS.get(int(l), f"Class_{l}") for l in labels]

        return SegmentationResult(
            boxes=boxes,
            masks=binary_masks,
            labels=label_names,
            scores=scores,
            num_instances=len(scores),
        )

    def visualize(
        self,
        image: np.ndarray,
        result: SegmentationResult,
        alpha: float = 0.4,
    ) -> np.ndarray:
        """
        Draw segmentation masks and bounding boxes on the image.

        Returns:
            (H, W, 3) uint8 visualization
        """
        if image.dtype != np.uint8:
            vis = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)
        else:
            vis = image.copy()

        if vis.ndim == 2:
            vis = np.stack([vis] * 3, axis=-1)

        # Color palette for different instances
        colors = [
            (255, 100, 100), (100, 255, 100), (100, 100, 255),
            (255, 255, 100), (255, 100, 255), (100, 255, 255),
            (255, 165, 0),   (147, 20,  255),
        ]

        overlay = vis.copy()

        for i, (box, mask, label, score) in enumerate(
            zip(result.boxes, result.masks, result.labels, result.scores)
        ):
            color = colors[i % len(colors)]

            # Draw mask overlay
            mask_area = mask > 0
            overlay[mask_area] = color

            # Draw bounding box
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # Label with confidence
            text = f"{label}: {score:.2f}"
            cv2.putText(vis, text, (x1, max(y1 - 10, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Blend overlay
        result_img = cv2.addWeighted(vis, 1 - alpha, overlay, alpha, 0)
        return result_img

    def compute_dice(self, pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
        """Compute Dice Similarity Coefficient between predicted and ground truth masks."""
        intersection = (pred_mask & gt_mask).sum()
        union = pred_mask.sum() + gt_mask.sum()
        if union == 0:
            return 1.0
        return 2.0 * intersection / union

    def result_to_dict(self, result: SegmentationResult) -> Dict:
        """Serialize SegmentationResult to JSON-compatible dict."""
        return {
            "num_instances": result.num_instances,
            "instances": [
                {
                    "label": label,
                    "score": round(float(score), 4),
                    "box": [round(float(v), 2) for v in box],
                    "mask_area_pixels": int(mask.sum()),
                }
                for label, score, box, mask in zip(
                    result.labels, result.scores, result.boxes, result.masks
                )
            ]
        }