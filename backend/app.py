"""
==============================================================================
AI Vision Screening: Spectacle Requirement Determination
FastAPI Backend REST Service & Web Interface Server
==============================================================================
Endpoints:
- GET /            : Serves the Web Application UI (or system status if UI not built)
- GET /status      : System health status & metadata JSON
- GET /api/status  : System health status & metadata JSON
- POST /predict    : Image prediction endpoint
- POST /api/predict: Image prediction endpoint
- GET /datasets    : Dataset summary JSON
- GET /api/datasets: Dataset summary JSON
==============================================================================
"""

import os
import io
import time
import base64
import json
from pathlib import Path
from typing import Optional
from PIL import Image

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd

from model import SpectacleVisionNet, predict_image_pipeline

# Initialize FastAPI App
app = FastAPI(
    title="AI Vision Screening - Spectacle Need Prediction API",
    description="Deep Neural Network API predicting whether a user requires spectacles based on ocular image screening.",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_DIR = Path(__file__).resolve().parent
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
MODEL_PATH = SAVED_MODELS_DIR / "spectacle_screening_model.pth"
METADATA_PATH = SAVED_MODELS_DIR / "model_metadata.json"
FRONTEND_DIST = BASE_DIR.parent / "frontend" / "dist"

# Mount Frontend static assets if available
if (FRONTEND_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global loaded model
loaded_model = None

def get_model():
    """Loads and caches the trained PyTorch Spectacle Vision Model."""
    global loaded_model
    if loaded_model is None:
        print("Initializing ResNet-18 SpectacleVisionNet model instance...")
        model = SpectacleVisionNet(num_classes=2, optical_feat_dim=4, pretrained=not MODEL_PATH.exists())
        
        if MODEL_PATH.exists():
            print(f"Loading trained weights from {MODEL_PATH}...")
            try:
                state_dict = torch.load(MODEL_PATH, map_location=device)
                model.load_state_dict(state_dict)
                print("Model weights loaded successfully.")
            except Exception as e:
                print(f"[Warning] Failed to load checkpoint ({e}). Running base model.")
        else:
            print(f"[Notice] No trained checkpoint found at {MODEL_PATH}. Initializing fresh model.")

        model.to(device)
        model.eval()
        loaded_model = model

    return loaded_model


@app.get("/reload")
@app.get("/api/reload")
@app.post("/reload")
@app.post("/api/reload")
def reload_model():
    """Forces reloading the newly trained ResNet-18 weights into memory."""
    global loaded_model
    import importlib
    import model as model_mod
    importlib.reload(model_mod)
    loaded_model = None
    m = get_model()
    return {
        "status": "success",
        "message": "ResNet-18 Transfer Learning model reloaded successfully.",
        "model_architecture": "ResNet-18 (Transfer Learning)"
    }


@app.get("/status")
@app.get("/api/status")
@app.get("/api")
def get_system_status():
    """Returns AI model and system diagnostic metadata."""
    has_weights = MODEL_PATH.exists()
    metadata = {}
    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, "r") as f:
                metadata = json.load(f)
        except Exception:
            pass

    return {
        "system": "AI-Based Vision Screening System",
        "status": "Online",
        "model_architecture": "ResNet-18 (Transfer Learning)",
        "trained_weights_loaded": has_weights,
        "primary_output": "Spectacle Requirement Determination (Spectacles Required / No Spectacles Needed)",
        "model_metadata": metadata,
        "endpoints": {
            "web_ui": "/",
            "predict": "POST /predict or POST /api/predict",
            "datasets": "GET /datasets or GET /api/datasets",
            "docs": "/docs"
        }
    }


@app.get("/")
def serve_root():
    """Serves the web application user interface directly."""
    index_file = FRONTEND_DIST / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return get_system_status()


@app.get("/datasets")
@app.get("/api/datasets")
def get_datasets_summary():
    """Returns information about the 3 indexed datasets and CSV files."""
    summary_path = BASE_DIR / "dataset" / "csv" / "dataset_summary.csv"
    if not summary_path.exists():
        summary_path = BASE_DIR.parent.parent / "dataset" / "csv" / "dataset_summary.csv"
    
    if summary_path.exists():
        df = pd.read_csv(summary_path)
        return {
            "total_images": int(df["image_count"].sum()),
            "datasets": df.to_dict(orient="records")
        }
    return {
        "message": "Dataset summary CSV not yet generated. Run create_dataset_csvs.py."
    }


@app.post("/predict")
@app.post("/api/predict")
async def predict_spectacles(
    file: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    symptom: Optional[str] = Form(None)
):
    """
    Primary screening endpoint:
    Accepts an uploaded image file (multipart/form-data) OR a base64 webcam string,
    along with an optional clinical vision symptom indicator.
    Returns whether the user needs spectacles or not.
    """
    start_time = time.time()

    # 1. Extract image bytes
    if file is not None:
        image_bytes = await file.read()
    elif image_base64:
        if "," in image_base64:
            image_base64 = image_base64.split(",", 1)[1]
        try:
            image_bytes = base64.b64decode(image_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 encoding: {str(e)}")
    else:
        raise HTTPException(status_code=400, detail="No image provided. Please upload an image file or provide image_base64.")

    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty image payload received.")

    # 2. Open with PIL
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted image format: {str(e)}")

    # 3. Predict with model & multimodal clinical symptom fusion
    model = get_model()
    result = predict_image_pipeline(model, pil_image, device=device, symptom=symptom)
    
    process_time_ms = round((time.time() - start_time) * 1000, 2)
    result["execution_time_ms"] = process_time_ms

    return result


if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI Backend & Web App on http://127.0.0.1:8000 ...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
