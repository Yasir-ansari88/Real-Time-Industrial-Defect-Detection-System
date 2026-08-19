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

        metrics = model.val()
        class_names = metrics.names
        per_class_mAP50 = {
            class_names[i] : float(metrics.box.ap50[i])
            for i in range(len(class_names))
        }

        result = {
            "config" : name,
            "params" : cfg,
            "mAP50_overall" : float(metrics.box.map50),
            "mAP50_95_overall" : float(metrics.box.map),
            "per_class_mAP50" : per_class_mAP50,
        }

        all_results.append(result)
        print(f"Results for {name} : mAP50 = {result['mAP50_overall']:.4f}",
            f"crazing = {per_class_mAP50.get('crazing','N/A'),}"
            f"rolled-in_scale = {per_class_mAP50.get('rolled-in_scale','N/A')},")
            

    all_results.sort(key=lambda r: r["mAP50_overall"], reverse=True)

    print(f"\n{'='*60}\nSUMMARY (ranked by overall mAP50)\n{'='*60}")
    for r in all_results:
        weak = r["per_class_mAP50"]
        print(
            f"{r['config']:20s} overall={r['mAP50_overall']:.4f}  "
            f"crazing={weak.get('crazing', 0):.4f}  "
            f"rolled-in_scale={weak.get('rolled-in_scale', 0):.4f}"
        )

    out_path = Path(args.project) / "tuning_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nFull results saved to {out_path}")

if __name__ == "__main__":
    main()