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


MODEL_PATH = os.environ.get("MODEL_PATH", "models/yolov8n_neu_final.pt")
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

class Detection(BaseModel):
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

class PredictionResponse(BaseModel):
    filename: str
    image_width: int
    image_height: int
    inference_ms: float
    detections: list[Detection]

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/metrics")
def metrics():
    return Response(content=render_metrics(), media_type="text/plain")

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    t0 = time.perf_counter()
    results = model.predict(image, conf=CONF_THRESHOLD, verbose=False)[0]
    inference_ms = (time.perf_counter() - t0) * 1000

    INFERENCE_LATENCY.observe(inference_ms / 1000)
    PREDICTION_COUNTER.inc()

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append(
            Detection(
                class_name=model.names[int(box.cls)],
                confidence=float(box.conf),
                x1=x1, y1=y1, x2=x2, y2=y2,
            )
        )

    return PredictionResponse(
        filename=file.filename or "unknown",
        image_width=image.width,
        image_height=image.height,
        inference_ms=round(inference_ms, 2),
        detections=detections,
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)