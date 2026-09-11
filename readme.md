# Real-Time Industrial Defect Detection System

## Computer Vision for Manufacturing Quality Control

A real-time computer vision pipeline designed for automated manufacturing quality inspection. The system processes live video feeds from assembly-line cameras, detects and classifies surface defects on manufactured parts, maps their locations, and can trigger automated sorting mechanisms.

The project is designed for **edge deployment**, enabling fast and consistent defect inspection directly on the production line.

---

## Project Overview

Manual visual inspection can be time-consuming, inconsistent, and difficult to scale. This project uses deep learning and computer vision to automate surface-defect detection.

The system is built around **YOLOv8 object detection** and processes images or live video streams to identify six major types of metal-surface defects.

### Supported Defect Classes

1. **Crazing**
2. **Inclusion**
3. **Patches**
4. **Pitted Surface**
5. **Rolled-in Scale**
6. **Scratches**

---

## Dataset

### NEU Metal Surface Defects Database

The primary dataset used for this project is the **NEU Metal Surface Defects Database**.

The dataset contains images of metal surfaces with six different categories of defects under varying visual conditions.

### Dataset Classes

| Class | Description |
|---|---|
| Crazing | Fine cracks appearing on the metal surface |
| Inclusion | Foreign material embedded in the metal |
| Patches | Irregular patch-like surface defects |
| Pitted Surface | Small pits or holes on the surface |
| Rolled-in Scale | Scale defects introduced during rolling |
| Scratches | Linear marks or scratches on the surface |


---

## System Architecture

```text
                 ┌──────────────────────┐
                 │   Camera / Video     │
                 │        Input         │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   OpenCV Processing  │
                 │  Frame Acquisition   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   YOLOv8 Detection   │
                 │   + Classification   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Defect Localization  │
                 │ & Bounding Boxes     │
                 └──────────┬───────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
     ┌─────────────────┐         ┌─────────────────┐
     │ Automated       │         │ API / Monitoring│
     │ Sorting Trigger │         │   Dashboard     │
     └─────────────────┘         └─────────────────┘
```

---

## Key Features

- Real-time surface-defect detection
- Detection and classification of six defect categories
- Bounding-box based defect localization
- Image and live-video processing
- Image preprocessing and augmentation
- YOLOv8-based object detection
- OpenCV camera/video integration
- ONNX/TensorRT optimization for edge deployment
- FastAPI or Flask backend
- Prometheus metrics and Grafana monitoring
- Designed for high-throughput manufacturing environments

---

## Technology Stack

### Programming Languages

- Python
- C++

### Computer Vision & Deep Learning

- PyTorch
- Ultralytics YOLOv8
- OpenCV
- Albumentations

### Edge Optimization

- ONNX Runtime
- NVIDIA TensorRT

### Backend

- FastAPI / Flask
- Uvicorn

### Monitoring

- Prometheus
- Grafana

### Version Control

- Git
- GitHub

---

## Project Workflow

### 1. Data Preparation

The NEU dataset is collected and organized into the required defect classes.

### 2. Image Preprocessing

Images are prepared for model training using computer-vision preprocessing techniques.

### 3. Data Augmentation

Augmentation techniques are applied to improve model robustness under varying lighting and surface conditions.

Example techniques include:

- Rotation
- Scaling
- Flipping
- Cropping
- Brightness/contrast adjustments

### 4. Model Training

YOLOv8 is trained on the prepared defect-detection dataset.

### 5. Model Evaluation

The trained model is evaluated using object-detection metrics such as:

- Precision
- Recall
- mAP@50
- mAP@50-95
- Confusion Matrix

### 6. Error Analysis

Incorrect detections and missed defects are analyzed to identify weaknesses in the model.

### 7. Edge Optimization

The trained model can be exported to ONNX and optimized using NVIDIA TensorRT or ONNX Runtime for faster inference on edge hardware.

### 8. Real-Time Detection

OpenCV captures frames from a camera or video stream and sends them to the optimized detection model.

### 9. Defect Localization

Detected defects are displayed using bounding boxes and class labels.

### 10. Automated Sorting

Detection results can be connected to a production-line sorting mechanism to separate defective products.

### 11. Monitoring

Prometheus and Grafana can be used to monitor system performance and inference-related metrics.

##  Getting Started

### 1. install dependencis
'''
pip install -r requirements.txt
'''
 
### 2. Clone / open the project
 
```powershell
cd "E:\Real-Time Industrial Defect Detection System"
```
 
### 3. Build and start all services
 
```powershell
docker compose -f docker/docker-compose.yml up --build
```
### 4. Verify everything is running
 
```powershell
docker ps
```
 
All three containers should show status `Up`.
 
## Accessing the Services
 
| Service          | URL                              | Notes                          |
|-------------------|-----------------------------------|---------------------------------|
| API Docs (Swagger)| http://localhost:8000/docs        | Test the `/predict` endpoint    |
| Raw Metrics       | http://localhost:8000/metrics     | Prometheus-format metrics       |
| Prometheus UI     | http://localhost:9090             | Query engine & scrape targets   |
| Prometheus Targets| http://localhost:9090/targets     | Confirm API scrape status = UP  |
| Grafana           | http://localhost:3000/login       | Default login: `admin` / `admin`|

## Testing the API
 
1. Open http://localhost:8000/docs
2. Expand the `/predict` endpoint
3. Click **Try it out**
4. Upload a test image from the `data/` folder
5. Click **Execute** and inspect the response — bounding boxes, class names, and confidence scores will be returned

## Setting Up the Grafana Dashboard
 
1. Log in to Grafana (`admin` / `admin`)
2. Go to **Connections → Data sources → Add data source → Prometheus**
3. Set the URL to `http://prometheus:9090` (use the Docker service name, not `localhost`, since Grafana and Prometheus communicate over the internal Docker network)
4. Click **Save & Test**
5. Create a new dashboard and add panels using the following metrics:
| Panel                         | PromQL Query                                                                                          |
|--------------------------------|--------------------------------------------------------------------------------------------------------|
| Total Defect Predictions       | `defect_detection_predictions_total`                                                                   |
| Average Inference Time         | `rate(defect_detection_inference_seconds_sum[5m]) / rate(defect_detection_inference_seconds_count[5m])`|
| Prediction Rate (per second)   | `rate(defect_detection_predictions_total[5m])`                                                         |
 
## Real-Time / Load Testing
 
A helper script (`realtime_test.py`) is included to continuously send test images to the API and simulate real-time traffic:
 
```powershell
python realtime_test.py
```
 
This sends a random image from the `data/` folder to `/predict` every 2 seconds, logging the response time and detection count to the console — useful for watching the Grafana dashboard update live.

---

