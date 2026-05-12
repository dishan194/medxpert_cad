"""
backend/models/report_generator.py

Simplified and stable AI medical report generator.
This version avoids heavy encoder-decoder crashes and generates
clean prescription-style radiology reports for demo purposes.
"""

from typing import Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Lightweight report generator for X-Ray and MRI analysis.
    Produces structured medical-style reports.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
    ):
        self.model_version = "2.0.0"
        logger.info("Simple Report Generator initialized.")

    def generate_structured_report(
        self,
        image_tensor,
        classification_result: Dict,
        segmentation_result: Optional[Dict] = None,
        modality: str = "Chest X-Ray",
    ) -> Dict:
        """
        Generate a clean medical report.
        """

        findings = classification_result.get("findings", [])
        is_normal = classification_result.get("is_normal", False)

        report_date = datetime.now().strftime("%d-%m-%Y %H:%M")

        # ---------------------------------------------------
        # NORMAL CASE
        # ---------------------------------------------------
        if is_normal or len(findings) == 0:

            findings_text = (
                "No significant abnormality detected.\n"
                "Cardiomediastinal silhouette is within normal limits.\n"
                "No focal consolidation or pleural effusion."
            )

            impression = (
                "Normal radiological appearance. "
                "No acute cardiopulmonary abnormality."
            )

            confidence = classification_result.get(
                "overall_confidence", 0.95
            )

        # ---------------------------------------------------
        # ABNORMAL CASE
        # ---------------------------------------------------
        else:

            findings_lines = []

            for finding in findings[:5]:

                pathology = finding.get("pathology", "Unknown")
                probability = finding.get("probability", 0.0)
                severity = finding.get("severity", "moderate")

                findings_lines.append(
                    f"- {pathology} detected "
                    f"(confidence: {probability:.1%}, severity: {severity})"
                )

            findings_text = "\n".join(findings_lines)

            top_pathology = findings[0].get("pathology", "abnormality")

            impression = (
                f"Imaging findings are suggestive of {top_pathology}. "
                f"Clinical correlation and specialist consultation recommended."
            )

            confidence = classification_result.get(
                "overall_confidence",
                findings[0].get("probability", 0.80)
            )

        # ---------------------------------------------------
        # SEGMENTATION DETAILS
        # ---------------------------------------------------
        segmentation_text = ""

        if segmentation_result:

            num_instances = segmentation_result.get(
                "num_instances", 0
            )

            if num_instances > 0:
                segmentation_text += (
                    f"\n\nSegmentation Analysis:\n"
                    f"{num_instances} abnormal region(s) identified."
                )

        # ---------------------------------------------------
        # FINAL REPORT
        # ---------------------------------------------------
        report_text = f"""
==================================================
            AI RADIOLOGY REPORT
==================================================

Date & Time : {report_date}

Modality    : {modality}

--------------------------------------------------
FINDINGS
--------------------------------------------------

{findings_text}

{segmentation_text}

--------------------------------------------------
IMPRESSION
--------------------------------------------------

{impression}

--------------------------------------------------
AI CONFIDENCE SCORE
--------------------------------------------------

{confidence:.1%}

--------------------------------------------------
DISCLAIMER
--------------------------------------------------

This AI-generated report is for educational/demo
purposes only and must be reviewed by a qualified
radiologist or physician.

==================================================
""".strip()

        return {
            "report_text": report_text,
            "findings_text": findings_text,
            "impression": impression,
            "confidence": round(confidence, 4),
            "model_version": self.model_version,
        }