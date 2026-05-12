import os
import uuid
import base64

from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse

from backend.services.inference_service import get_inference_service

router = APIRouter()

# Create folder for heatmaps
HEATMAP_DIR = "static/heatmaps"
os.makedirs(HEATMAP_DIR, exist_ok=True)


@router.post("/analyze/xray")
async def analyze_xray(
    file: UploadFile = File(...),
    run_segmentation: bool = Query(True),
    session_id: str = Query(None)
):

    try:

        # Generate session ID
        if not session_id:
            session_id = str(uuid.uuid4())

        # Read uploaded image
        contents = await file.read()

        # Get inference service
        inference_service = get_inference_service()

        # Run AI prediction
        result = inference_service.analyze_xray(
            image_bytes=contents,
            run_segmentation=run_segmentation
        )

        # ---------------------------------------
        # SAVE HEATMAP IMAGE
        # ---------------------------------------

        heatmap_filename = None
        heatmap_url = None

        if result.get("heatmap_base64"):

            heatmap_data = base64.b64decode(
                result["heatmap_base64"]
            )

            heatmap_filename = f"{session_id}.png"

            heatmap_path = os.path.join(
                HEATMAP_DIR,
                heatmap_filename
            )

            with open(heatmap_path, "wb") as f:
                f.write(heatmap_data)

            heatmap_url = f"/static/heatmaps/{heatmap_filename}"

        # ---------------------------------------
        # CLEAN HUMAN READABLE RESPONSE
        # ---------------------------------------

        clean_findings = []

        findings = result["classification"]["findings"]

        for item in findings:

            clean_findings.append({
                "condition": item["pathology"],
                "confidence": round(
                    item["probability"] * 100,
                    2
                ),
                "severity": item["severity"]
            })

        response = {
            "success": True,
            "session_id": session_id,
            "modality": "xray",

            "summary": {
                "model": result["classification"]["model_type"],
                "overall_confidence": round(
                    result["classification"]["overall_confidence"] * 100,
                    2
                ),
                "is_normal": result["classification"]["is_normal"]
            },

            "findings": clean_findings,

            "heatmap_url": heatmap_url,

            "message": "X-ray analysis completed successfully"
        }

        return JSONResponse(content=response)

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )