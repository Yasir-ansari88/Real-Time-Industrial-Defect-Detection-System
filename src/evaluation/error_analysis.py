import argparse
import json
from pathlib import Path 
from ultralytics import YOLO
def load_yolo_label(label_path:Path):
    """Returns list of(class_id, cx, cy, w, h) - normalized YOLO format."""
    boxes=[]
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        parts=line.split()
        boxes.append((int(parts[0]), *map(float, parts[1:5])))
    return boxes

def yolo_to_xyxy(cx,cy,w,h,img_w,img_h):
    x1=(cx-w/2)*img_w
    y1=(cy-h/2)*img_h
    x2=(cx+w/2)*img_w
    y2=(cy+h/2)*img_h
    return x1,y1,x2,y2

def iou(box_a,box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--data", type=str, default="data/processed/data.yaml")
    parser.add_argument("--images-dir", type=str, default="data/processed/images/val")
    parser.add_argument("--labels-dir", type=str, default="data/processed/labels/val")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for predictions")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold to count a match as correct")
    parser.add_argument("--out-dir", type=str, default="models/runs/error_analysis")
    parser.add_argument("--weak-classes", type=str, nargs="+",default=["crazing", "rolled-in_scale"])
    args = parser.parse_args()

    out_dir=Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    class_names=model.names  # idx -> name
    print("Running model.val() for confusion matrix + standard metrics...")
    metrics=model.val(data=args.data, conf=args.conf, iou=args.iou,project=str(out_dir), name="val_confusion")
    print(f"Confusion matrix + PR curves saved under {out_dir}/val_confusion/")







    
     




