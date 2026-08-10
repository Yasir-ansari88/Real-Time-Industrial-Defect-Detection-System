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

---

