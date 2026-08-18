import argparse
import json
from pathlib import Path
from ultralytics import YOLO

CONFIGS = [
    {
        "name" : "baseline",
        "lr0" : 0.01,
        "imgsz" : 640,
        "degrees" : 25,
        "weight_decay" : 0.0005,
    },
    {
        "name" : "lower_lr",
        "lr0" : 0.005,
        "imgsz" : 640,
        "degrees" : 25,
        "weight_decay" : 0.0005, 
    },
    {
        "name" : "larger_imgsz",
        "lr0" : 0.01,
        "imgsz" : 832,
        "degrees" : 25,
        "weight_decay" : 0.0005,   
    },
    {
        "name" : "gentler_rotation",
        "lr0" : 0.01,
        "imgsz" : 640,
        "degrees" : 10,
         "weight_decay" : 0.0005,
    }
]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default="data/processed/data.ymal")
    parser.add_argument("--tune-epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", type=str, default="models/runs/tune")
    parser.add_argument("--base-model", type=str, default="yolov8n.pt")
    args = parser.parse_args()

    all_results = []

    for cfg in CONFIGS:
        name = cfg["name"]
        print(f"\n{'='*60}\nTraining config: {name} -> {cfg}\n{'='*60}")

        model = YOLO(args.base_model)
        model.train(
            data = args.data,
            epochs = args.tune_epochs,
            imgsz = cfg["imgsz"],
            batch = args.batch,
            lr0 = cfg["lr0"],
            degrees = cfg["degrees"],
            weight_decay = cfg["weight_decay"],
            project = args.project,
            name = name,
            patience = args.tune_epochs,
            verbose = False,
        )
