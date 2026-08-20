import argparse
import json
from pathlib import Path 
from ultralytics import YOLO
def load_yolo_labels(label_path:Path):
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
     
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    image_paths = sorted(images_dir.glob("*.jpg"))

    fn_records = []  # missed ground-truth boxes
    fp_records = []  # unmatched predictions
    print(f"\nScanning {len(image_paths)} validation images for FP/FN...")
    for img_path in image_paths:
        gt_boxes_raw = load_yolo_labels(labels_dir / f"{img_path.stem}.txt")
        if not gt_boxes_raw:
            continue

        result = model.predict(str(img_path), conf=args.conf, verbose=False)[0]
        img_h, img_w = result.orig_shape
        gt_boxes = [
            {"cls": cls, "box": yolo_to_xyxy(cx, cy, w, h, img_w, img_h), "matched": False}
            for cls, cx, cy, w, h in gt_boxes_raw
        ]
        pred_boxes = [
            {
                "cls": int(box.cls.item()),
                "conf": float(box.conf.item()),
                "box": tuple(box.xyxy[0].tolist()),
                "matched": False,
            }
            for box in result.boxes
        ]
        for pred in pred_boxes:
            best_iou, best_gt = 0.0, None
            for gt in gt_boxes:
                if gt["matched"] or gt["cls"] != pred["cls"]:
                    continue
                cur_iou = iou(pred["box"], gt["box"])
                if cur_iou > best_iou:
                    best_iou, best_gt = cur_iou, gt
            if best_gt is not None and best_iou >= args.iou:
                best_gt["matched"] = True
                pred["matched"] = True
                for pred in pred_boxes:
                 if not pred["matched"]:
                   fp_records.append({
                     "image": img_path.name,
                     "class": class_names[pred["cls"]],
                     "confidence": round(pred["conf"], 3),
                })
                for gt in gt_boxes:
                 if not gt["matched"]:
                   fn_records.append({
                    "image": img_path.name,
                    "class": class_names[gt["cls"]],
                })
    # 3. Report
    print(f"\nTotal false positives: {len(fp_records)}")
    print(f"Total false negatives (missed detections): {len(fn_records)}")

    from collections import Counter
    fp_by_class = Counter(r["class"] for r in fp_records)
    fn_by_class = Counter(r["class"] for r in fn_records)

    print("\nFalse positives by class:")
    for cls, count in fp_by_class.most_common():
        print(f"  {cls}: {count}")

    print("\nFalse negatives (missed) by class:")
    for cls, count in fn_by_class.most_common():
        print(f"  {cls}: {count}")

    print("\n--- Focus: weak classes ---")
    for cls in args.weak_classes:
        print(f"\n{cls}:")
        print(f"  False positives: {fp_by_class.get(cls, 0)}")
        print(f"  False negatives (missed): {fn_by_class.get(cls, 0)}")
        worst_images = sorted(
            {r["image"] for r in fn_records if r["class"] == cls}
        )[:10]
        if worst_images:
            print(f"  Sample images with missed {cls} detections (inspect these):")
            for img in worst_images:
                print(f"    - {img}")
    # Save full records for later inspection
    report = {
        "false_positives": fp_records,
        "false_negatives": fn_records,
        "fp_by_class": dict(fp_by_class),
        "fn_by_class": dict(fn_by_class),
    }
    report_path = out_dir / "error_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nFull FP/FN report saved to {report_path}")
    print("\nNext: open the sample images listed above for weak classes and check "
          "whether the defect is genuinely hard to see, mislabeled, or too small "
          "relative to image size — that tells you whether to add more training "
          "data, adjust augmentation, or increase image resolution.")


if __name__ == "__main__":
    main()





   

        













    
     




