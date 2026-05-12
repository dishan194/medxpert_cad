"""
backend/models/grad_cam.py
Gradient-weighted Class Activation Mapping (Grad-CAM) for explainability.
Produces heatmaps showing which regions the model focused on.
Reference: Selvaraju et al., 2017 — "Grad-CAM: Visual Explanations from Deep Networks"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from typing import Optional, List, Tuple
import logging

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Grad-CAM implementation for CNN-based models.
    Hooks into the last convolutional layer to capture gradients and activations.
    """

    def __init__(self, model: nn.Module, target_layer: Optional[nn.Module] = None):
        """
        Args:
            model: PyTorch model (ResNet50, VGG16, etc.)
            target_layer: The layer to hook. If None, auto-detects last conv layer.
        """
        self.model = model
        self.model.eval()

        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._hooks = []

        # Auto-detect target layer if not specified
        if target_layer is None:
            target_layer = self._find_last_conv(model)

        if target_layer is None:
            raise ValueError("Could not find a convolutional layer. Specify target_layer explicitly.")

        self._register_hooks(target_layer)
        logger.info(f"Grad-CAM hooked on: {target_layer.__class__.__name__}")

    def _find_last_conv(self, model: nn.Module) -> Optional[nn.Module]:
        """Find the last Conv2d layer in the model."""
        last_conv = None
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                last_conv = module
        return last_conv

    def _register_hooks(self, layer: nn.Module):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hooks.append(layer.register_forward_hook(forward_hook))
        self._hooks.append(layer.register_full_backward_hook(backward_hook))

    def generate(
        self,
        image_tensor: torch.Tensor,
        target_class: int,
        original_image_size: Tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a specific class.

        Args:
            image_tensor: (1, C, H, W) preprocessed image
            target_class: Class index to explain
            original_image_size: (H, W) to resize heatmap to

        Returns:
            heatmap: (H, W) numpy array in [0, 1]
        """
        self.model.zero_grad()

        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.requires_grad_(True)

        # Forward pass
        output = self.model(image_tensor)
        if isinstance(output, tuple):
            output = output[0]  # (logits, features) — take logits

        # Backward for target class
        class_score = output[0, target_class]
        class_score.backward()

        # Grad-CAM computation
        gradients = self.gradients   # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)

        # Global average pooling of gradients
        weights = gradients.mean(dim=[2, 3], keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activations
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)  # ReLU — only positive contributions

        # Normalize to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to original image size
        heatmap = cv2.resize(cam, (original_image_size[1], original_image_size[0]))

        return heatmap

    def generate_multi_label(
        self,
        image_tensor: torch.Tensor,
        class_indices: List[int],
        original_image_size: Tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        """
        Generate combined Grad-CAM for multiple classes (multi-label X-ray).
        Averages heatmaps weighted by class probability.
        """
        combined = np.zeros(original_image_size, dtype=np.float32)

        for class_idx in class_indices:
            heatmap = self.generate(image_tensor.clone(), class_idx, original_image_size)
            combined += heatmap

        if len(class_indices) > 0:
            combined /= len(class_indices)

        return np.clip(combined, 0, 1)

    def overlay_on_image(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay heatmap on the original image.

        Args:
            original_image: (H, W, 3) uint8 RGB image
            heatmap: (H, W) float32 in [0, 1]
            alpha: Blend weight for heatmap
            colormap: OpenCV colormap

        Returns:
            overlaid: (H, W, 3) uint8 image
        """
        # Convert heatmap to colorized version
        heatmap_uint8 = (heatmap * 255).astype(np.uint8)
        colored = cv2.applyColorMap(heatmap_uint8, colormap)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

        # Ensure original is uint8 RGB
        if original_image.dtype != np.uint8:
            orig = ((original_image - original_image.min()) /
                    (original_image.max() - original_image.min() + 1e-8) * 255).astype(np.uint8)
        else:
            orig = original_image.copy()

        if orig.ndim == 2:
            orig = np.stack([orig] * 3, axis=-1)

        # Resize colored if needed
        if colored.shape[:2] != orig.shape[:2]:
            colored = cv2.resize(colored, (orig.shape[1], orig.shape[0]))

        # Blend
        overlaid = cv2.addWeighted(orig, 1 - alpha, colored, alpha, 0)
        return overlaid

    def to_base64(self, image: np.ndarray) -> str:
        """Convert numpy image to base64 string for API response."""
        import base64
        from PIL import Image
        import io

        pil_img = Image.fromarray(image.astype(np.uint8))
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def cleanup(self):
        """Remove hooks to free memory."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __del__(self):
        self.cleanup()


class GradCAMPlusPlus(GradCAM):
    """
    Grad-CAM++ (improved variant with better localization).
    Reference: Chattopadhay et al., 2018
    """

    def generate(
        self,
        image_tensor: torch.Tensor,
        target_class: int,
        original_image_size: Tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        self.model.zero_grad()

        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.requires_grad_(True)

        output = self.model(image_tensor)
        if isinstance(output, tuple):
            output = output[0]

        class_score = output[0, target_class]
        class_score.backward()

        gradients = self.gradients    # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)

        # Grad-CAM++ weight computation
        grad_sq = gradients ** 2
        grad_cube = gradients ** 3
        sum_act = activations.sum(dim=[2, 3], keepdim=True)
        alpha = grad_sq / (2 * grad_sq + sum_act * grad_cube + 1e-7)
        alpha = alpha * (gradients > 0).float()

        weights = (alpha * F.relu(gradients)).mean(dim=[2, 3], keepdim=True)

        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cv2.resize(cam, (original_image_size[1], original_image_size[0]))


def build_gradcam(model: nn.Module, model_type: str = "resnet50") -> GradCAM:
    """
    Factory to build Grad-CAM with correct target layer for each architecture.
    """
    if model_type == "resnet50":
        # Last residual block's last conv layer
        try:
            target = model.feature_extractor[-1][-1].conv3
        except (AttributeError, IndexError):
            target = None
        return GradCAMPlusPlus(model, target_layer=target)

    elif model_type == "vgg16":
        try:
            target = model.features[-1]
        except (AttributeError, IndexError):
            target = None
        return GradCAM(model, target_layer=target)

    else:
        return GradCAM(model)