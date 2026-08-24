import argparse
import pathlib as Path

from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=str, required=True, help="Path to trained .pt weights")
    parser.add_argument("--format", type=str, choices=["onnx", "tensorrt", "both"], default="onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--half", action="store_true", help="Export in FP16")
    args = parser.parse_args()

    weight_path = Path(args.weights)
    if not weight_path.exists():
        raise FileNotFoundError(f"{weight_path} not found")

    model = YOLO(str(weight_path))

    if args.format in ("onnx", "both"):
        print("Exporting to Onnx")
        onnx_path = model.export(format="onnx", imgsz=args.imgsz, half=args.half, simplify=False)
        print(f"Onnx model saved to : {onnx_path}")

    if args.format in ("tensorrt", "both"):
        print("Export to TensoRT engin")
        engin_path = model.export(format="engine", imgsz=args.imgsz, half=args.half)
        print(f"TensoRT Engine saved to : {engin_path}")

if __name__ == "__main__":
    main()