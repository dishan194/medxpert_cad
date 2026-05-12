"""
tests/unit/test_models.py
Unit tests for all model components.
Run: pytest tests/ -v --cov=backend
"""

import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch


# ─────────────────────────────────────────────────────────────
# DICOM Utils Tests
# ─────────────────────────────────────────────────────────────

class TestDICOMProcessor:
    """Tests for DICOMProcessor utility."""

    def setup_method(self):
        from backend.utils.dicom_utils import DICOMProcessor
        self.proc = DICOMProcessor(target_size=(224, 224))

    def test_normalize_xray_output_range(self):
        """Normalized X-ray should be uint8 in [0, 255]."""
        dummy = np.random.randint(0, 65535, (512, 512), dtype=np.uint16).astype(np.float32)
        result = self.proc.normalize_xray(dummy)
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_to_rgb_grayscale(self):
        """2D grayscale should become 3-channel."""
        gray = np.random.randint(0, 255, (224, 224), dtype=np.uint8)
        rgb  = self.proc.to_rgb(gray)
        assert rgb.shape == (224, 224, 3)
        np.testing.assert_array_equal(rgb[:, :, 0], rgb[:, :, 1])

    def test_resize_output_shape(self):
        """Resize should produce target_size output."""
        image = np.random.randint(0, 255, (512, 400, 3), dtype=np.uint8)
        resized = self.proc.resize(image)
        assert resized.shape[:2] == (224, 224)

    def test_preprocess_for_model_shape(self):
        """Preprocessed tensor should be (3, H, W) float32."""
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        result = self.proc.preprocess_for_model(image)
        assert result.shape == (3, 224, 224)
        assert result.dtype == np.float32

    def test_apply_windowing_clips_values(self):
        """Windowing should clip to [0, 255]."""
        pixels = np.arange(-1000, 1000, dtype=np.float32)
        windowed = self.proc.apply_windowing(pixels, center=40, width=80)
        assert windowed.min() >= 0
        assert windowed.max() <= 255


# ─────────────────────────────────────────────────────────────
# X-Ray Classifier Tests
# ─────────────────────────────────────────────────────────────

class TestResNet50Classifier:
    """Tests for ResNet50 multi-label classifier."""

    def setup_method(self):
        from backend.models.xray_classifier import ResNet50Classifier
        self.model = ResNet50Classifier(num_classes=14, pretrained=False)
        self.model.eval()

    def test_forward_output_shapes(self):
        """Model should return (B, 14) logits and (B, 2048, 7, 7) features."""
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits, features = self.model(x)
        assert logits.shape == (2, 14)
        assert features.shape[1] == 2048

    def test_probabilities_range(self):
        """Probabilities from sigmoid should be in [0, 1]."""
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            probs = self.model.get_probabilities(x)
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_batch_consistency(self):
        """Same image twice in batch should produce same results."""
        x = torch.randn(1, 3, 224, 224)
        batch = x.repeat(2, 1, 1, 1)
        with torch.no_grad():
            logits, _ = self.model(batch)
        torch.testing.assert_close(logits[0], logits[1], atol=1e-5, rtol=1e-5)


class TestXRayClassifierWrapper:
    """Tests for XRayClassifier high-level wrapper."""

    def setup_method(self):
        from backend.models.xray_classifier import XRayClassifier
        self.classifier = XRayClassifier(device="cpu", threshold=0.5)

    def test_predict_returns_dict(self):
        """predict() should return a dict with required keys."""
        x = torch.randn(1, 3, 224, 224)
        result = self.classifier.predict(x)
        assert "is_normal" in result
        assert "findings" in result
        assert "all_scores" in result
        assert "overall_confidence" in result
        assert isinstance(result["findings"], list)

    def test_all_scores_keys(self):
        """all_scores should have one entry per pathology."""
        x = torch.randn(1, 3, 224, 224)
        result = self.classifier.predict(x)
        assert len(result["all_scores"]) == 14
        from backend.models.xray_classifier import XRayClassifier
        for label in XRayClassifier.PATHOLOGY_LABELS:
            assert label in result["all_scores"]

    def test_findings_sorted_by_probability(self):
        """Findings should be sorted descending by probability."""
        x = torch.randn(3, 3, 224, 224)
        result = self.classifier.predict(x)
        findings = result["findings"]
        if len(findings) >= 2:
            probs = [f["probability"] for f in findings]
            assert probs == sorted(probs, reverse=True)


# ─────────────────────────────────────────────────────────────
# MRI Classifier Tests
# ─────────────────────────────────────────────────────────────

class TestMRIClassifier:
    """Tests for MRI tumor classifier."""

    def setup_method(self):
        from backend.models.mri_classifier import MRIClassifier
        self.classifier = MRIClassifier(device="cpu")

    def test_predict_returns_class(self):
        """predict() should return a valid tumor class."""
        from backend.models.mri_classifier import MRIClassifier
        x = torch.randn(1, 3, 224, 224)
        result = self.classifier.predict(x)
        assert result["predicted_class"] in MRIClassifier.TUMOR_LABELS
        assert 0.0 <= result["confidence"] <= 1.0

    def test_probabilities_sum_to_one(self):
        """Softmax probabilities should sum to 1."""
        x = torch.randn(1, 3, 224, 224)
        result = self.classifier.predict(x)
        total = sum(result["all_probabilities"].values())
        assert abs(total - 1.0) < 1e-4


# ─────────────────────────────────────────────────────────────
# Grad-CAM Tests
# ─────────────────────────────────────────────────────────────

class TestGradCAM:
    """Tests for Grad-CAM explainability."""

    def setup_method(self):
        from backend.models.xray_classifier import ResNet50Classifier
        from backend.models.grad_cam import GradCAM
        self.model = ResNet50Classifier(num_classes=14, pretrained=False)
        self.gradcam = GradCAM(self.model)

    def test_heatmap_shape(self):
        """Heatmap should match specified output size."""
        x = torch.randn(1, 3, 224, 224)
        heatmap = self.gradcam.generate(x, target_class=0, original_image_size=(512, 512))
        assert heatmap.shape == (512, 512)

    def test_heatmap_range(self):
        """Heatmap values should be in [0, 1]."""
        x = torch.randn(1, 3, 224, 224)
        heatmap = self.gradcam.generate(x, target_class=3, original_image_size=(224, 224))
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0

    def test_overlay_shape(self):
        """Overlay should have same shape as original image."""
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        heatmap = np.random.rand(224, 224).astype(np.float32)
        overlaid = self.gradcam.overlay_on_image(image, heatmap)
        assert overlaid.shape == (224, 224, 3)


# ─────────────────────────────────────────────────────────────
# Report Generator Tests
# ─────────────────────────────────────────────────────────────

class TestReportGenerator:
    """Tests for clinical report generation."""

    def setup_method(self):
        from backend.models.report_generator import ReportGenerator
        self.generator = ReportGenerator(device="cpu")

    def test_bleu_returns_scores(self):
        """BLEU evaluation should return dict with bleu1, bleu2, bleu4."""
        scores = self.generator.compute_bleu(
            references=["The lung shows no evidence of focal consolidation."],
            hypothesis="No consolidation is seen in the lungs.",
        )
        assert "bleu1" in scores
        assert "bleu4" in scores
        assert 0.0 <= scores["bleu4"] <= 1.0

    def test_structured_report_format(self):
        """Generated report should contain key sections."""
        x = torch.randn(1, 3, 224, 224)
        classification = {
            "is_normal": False,
            "findings": [{"pathology": "Pneumonia", "probability": 0.87, "severity": "high"}],
            "overall_confidence": 0.87,
        }
        result = self.generator.generate_structured_report(
            x, classification, modality="Chest X-Ray"
        )
        assert "report_text" in result
        assert "findings_text" in result
        assert "impression" in result
        assert "RADIOLOGY REPORT" in result["report_text"]
        assert "Pneumonia" in result["findings_text"]

    def test_normal_report_text(self):
        """Normal study should produce appropriate impression."""
        x = torch.randn(1, 3, 224, 224)
        classification = {"is_normal": True, "findings": [], "overall_confidence": 0.95}
        result = self.generator.generate_structured_report(x, classification)
        assert "No significant abnormality" in result["impression"] or \
               "Normal" in result["impression"]


# ─────────────────────────────────────────────────────────────
# Audit Logger Tests
# ─────────────────────────────────────────────────────────────

class TestAuditLogger:
    """Tests for HIPAA-compliant audit logging."""

    def setup_method(self):
        from backend.utils.audit_logger import AuditLogger
        self.logger = AuditLogger(log_path="/tmp/test_audit.log", enabled=True)

    def test_user_id_hashed(self):
        """User IDs should be hashed, not stored in plain text."""
        from backend.utils.audit_logger import AuditEventType
        record = self.logger._build_record(
            event_type=AuditEventType.AUTH_SUCCESS,
            session_id="sess-123",
            user_id="dr.smith@hospital.com",
            ip_address="10.0.0.1",
            resource_id=None,
        )
        # The hash should NOT contain the original email
        assert "dr.smith@hospital.com" not in str(record.get("user_id_hash", ""))
        assert len(record["user_id_hash"]) == 16  # Truncated SHA256

    def test_log_does_not_raise(self):
        """Logging should not raise exceptions."""
        from backend.utils.audit_logger import AuditEventType
        self.logger.log(AuditEventType.IMAGE_UPLOAD, session_id="test", ip_address="127.0.0.1")
        self.logger.log_upload("test-session", "10.0.0.1", "xray", 1024 * 1024)


# ─────────────────────────────────────────────────────────────
# Vocabulary Tests
# ─────────────────────────────────────────────────────────────

class TestMedicalVocabulary:
    """Tests for report tokenizer vocabulary."""

    def setup_method(self):
        from backend.models.report_generator import MedicalVocabulary
        self.vocab = MedicalVocabulary()
        for word in ["the", "lung", "shows", "consolidation", "cardiomegaly", "normal"]:
            self.vocab.add_word(word)

    def test_special_tokens_present(self):
        """Special tokens should be indexed starting at 0."""
        assert self.vocab.word2idx["<PAD>"] == 0
        assert self.vocab.word2idx["<SOS>"] == 1
        assert self.vocab.word2idx["<EOS>"] == 2
        assert self.vocab.word2idx["<UNK>"] == 3

    def test_encode_decode_roundtrip(self):
        """Encoding then decoding should recover original words."""
        text = "the lung shows consolidation"
        ids  = self.vocab.encode(text)
        decoded = self.vocab.decode(ids)
        for word in text.split():
            assert word in decoded

    def test_unknown_word_handling(self):
        """Unknown words should map to <UNK> index."""
        ids = self.vocab.encode("xyzzy unknown_term_12345")
        unk_idx = self.vocab.word2idx["<UNK>"]
        assert unk_idx in ids