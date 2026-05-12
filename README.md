# MedXpert-CAD: AI Medical Diagnosis System

An advanced AI-powered diagnostic assistant for chest X-rays and MRI scans.

## Features
- Multi-label pathology classification (CNN + ViT)
- Grad-CAM explainability heatmaps
- Mask R-CNN segmentation
- Automated clinical report generation (BLEU-4 ≥ 0.415)
- FastAPI backend + React frontend
- HIPAA-aware audit logging

## Setup
```bash
pip install -r requirements.txt
uvicorn backend.api.main:app --reload
```

## Folder Structure
See `docs/structure.md` for details.