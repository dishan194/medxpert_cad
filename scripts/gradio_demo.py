"""
scripts/gradio_demo.py
Gradio-based interactive demo for MedXpert-CAD.
Provides a polished UI for uploading and analyzing medical images.

Usage:
    python scripts/gradio_demo.py
"""

import base64
import io
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def analyze_image(image, modality: str, run_segmentation: bool):
    """Main Gradio callback function."""
    if image is None:
        return None, None, "Please upload an image.", "{}"

    try:
        from backend.services.inference_service import get_inference_service

        # Convert PIL/numpy to bytes
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(image.astype(np.uint8))
        else:
            pil_img = image

        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        file_bytes = buf.getvalue()

        service = get_inference_service()

        if modality == "Chest X-Ray":
            result = service.analyze_xray(file_bytes, run_segmentation=run_segmentation)
        else:
            result = service.analyze_mri(file_bytes, run_segmentation=run_segmentation)

        # Decode heatmap
        heatmap_img = None
        if result.get("heatmap_base64"):
            heatmap_bytes = base64.b64decode(result["heatmap_base64"])
            heatmap_img = Image.open(io.BytesIO(heatmap_bytes))

        seg_img = None
        if result.get("segmentation_vis_base64"):
            seg_bytes = base64.b64decode(result["segmentation_vis_base64"])
            seg_img = Image.open(io.BytesIO(seg_bytes))

        report_text = result["report"]["report_text"]
        findings_json = _format_findings(result)

        return heatmap_img, seg_img, report_text, findings_json

    except ImportError as e:
        return None, None, f"Model import error: {e}", "{}"
    except Exception as e:
        return None, None, f"Analysis failed: {e}", "{}"


def _format_findings(result: dict) -> str:
    """Format classification results as readable text."""
    import json
    clf = result["classification"]
    output = {
        "is_normal": clf.get("is_normal", True),
        "findings":  clf.get("findings", []),
        "inference_time_ms": result.get("inference_time_ms", 0),
    }
    if "predicted_class" in clf:
        output["predicted_class"] = clf["predicted_class"]
        output["confidence"] = clf.get("confidence", 0)
    return json.dumps(output, indent=2)


def create_demo():
    import gradio as gr

    with gr.Blocks(
        title="MedXpert-CAD — AI Medical Diagnosis",
        theme=gr.themes.Soft(primary_hue="blue"),
        css="""
            .title-row { text-align: center; margin-bottom: 20px; }
            .warning-box { background: #fff3cd; border: 1px solid #ffc107;
                           padding: 10px; border-radius: 6px; margin: 10px 0; }
        """,
    ) as demo:

        gr.HTML("""
            <div class='title-row'>
                <h1>🩺 MedXpert-CAD</h1>
                <h3>AI-Powered Medical Diagnostic Assistant</h3>
                <p>Multi-label classification · Grad-CAM explainability · Segmentation · Report generation</p>
            </div>
        """)

        gr.HTML("""
            <div class='warning-box'>
                ⚠️ <strong>Clinical Disclaimer:</strong> This AI tool is for research and educational purposes only.
                All results must be reviewed and validated by a licensed radiologist before clinical use.
            </div>
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Upload Image")
                image_input = gr.Image(
                    label="Medical Image (JPEG/PNG/DICOM)",
                    type="numpy",
                    height=300,
                )
                modality = gr.Radio(
                    choices=["Chest X-Ray", "MRI Brain"],
                    value="Chest X-Ray",
                    label="Modality",
                )
                run_seg = gr.Checkbox(value=True, label="Run Mask R-CNN Segmentation")
                analyze_btn = gr.Button("🔬 Analyze", variant="primary", size="lg")

                gr.Markdown("### 📊 Classification Results")
                findings_output = gr.JSON(label="Findings (JSON)")

            with gr.Column(scale=2):
                gr.Markdown("### 🔥 Grad-CAM Heatmap")
                heatmap_output = gr.Image(
                    label="Attention Heatmap (regions influencing diagnosis)",
                    height=300,
                )

                gr.Markdown("### 🗺️ Segmentation Overlay")
                seg_output = gr.Image(
                    label="Mask R-CNN Segmentation",
                    height=300,
                )

                gr.Markdown("### 📋 Clinical Report")
                report_output = gr.Textbox(
                    label="Generated Radiology Report",
                    lines=20,
                    show_copy_button=True,
                )

        analyze_btn.click(
            fn=analyze_image,
            inputs=[image_input, modality, run_seg],
            outputs=[heatmap_output, seg_output, report_output, findings_output],
        )

        gr.Markdown("""
        ---
        **Model Architecture:** ResNet50 → Grad-CAM++ → Mask R-CNN → LSTM Report Generator
        **Dataset:** NIH ChestX-ray14 (14 pathologies) | IU X-Ray (report generation)
        **Performance Targets:** AUC ≥ 0.85 | BLEU-4 ≥ 0.415 | Dice ≥ 95%
        """)

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )