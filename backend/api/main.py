from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes.analysis import router as analysis_router

app = FastAPI(
    title="MedXpert-CAD API",
    description="AI-powered Medical Diagnosis System for X-ray & MRI",
    version="1.0.0"
)

# CORS (for React frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(analysis_router, prefix="/api/v1")

@app.get("/")
def root():
    return {
        "success": True,
        "message": "MedXpert-CAD API is running successfully",
        "version": "1.0.0",
        "docs": "/docs"
    }