import io
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel
from ultralytics import YOLO

from monitoring.metrices import track_uptime_start, PREDICTION_COUNTER, INFERENCE_LATENCY, render_metrics


MODEL_PATH = os.environ.get("MODEL_PATH", "E:\Real-Time Industrial Defect Detection System\models\yolov8n_neu_final.pt")
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.25"))

model = None
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model weights not found at {MODEL_PATH}."
            f"environment variable or place your final model there."
        )
    model = YOLO(MODEL_PATH)
    track_uptime_start()
    print(f"Model loaded from {MODEL_PATH}")
    yield

app = FastAPI(
    title="Industrial Defect Detection API",
    description="Real-time surface defect detection for manufacturing quality control.",
    version="1.0.0",
    lifespan=lifespan,
)