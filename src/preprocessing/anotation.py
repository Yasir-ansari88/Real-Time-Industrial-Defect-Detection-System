import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
import yaml
from tqdm import tqdm

CLASSES=["crazing","inclusion","patches","pitted_surface","rolled-in_scale","scratches"]
CLASS_TO_IDX={name:i for i, name in enumerate(CLASSES)}
def voc_to_yolo_bbox(xmin, ymin, xmax, ymax, img_w, img_h):
    """ normalized 0-1."""
    cx = ((xmin + xmax) / 2) / img_w
    cy = ((ymin + ymax) / 2) / img_h
    w = (xmax - xmin) / img_w
    h = (ymax - ymin) / img_h
    return cx, cy, w, h

def parse_voc_annotation(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename")
    size = root.find("size")
    img_w = int(size.findtext("width"))
    img_h = int(size.findtext("height"))
    yolo_lines = []
    for obj in root.findall("object"):
        cls_name = obj.findtext("name").strip().lower().replace(" ", "_")
        if cls_name not in CLASS_TO_IDX:
            print(f"  [warn] unknown class '{cls_name}' in {xml_path.name}, skipping object")
            continue
        cls_idx = CLASS_TO_IDX[cls_name]

        bnd = obj.find("bndbox")
        xmin = float(bnd.findtext("xmin"))
        ymin = float(bnd.findtext("ymin"))
        xmax = float(bnd.findtext("xmax"))
        ymax = float(bnd.findtext("ymax"))

        cx, cy, w, h = voc_to_yolo_bbox(xmin, ymin, xmax, ymax, img_w, img_h)
        yolo_lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return filename, yolo_lines

def find_image_for_annotation(images_dir: Path, stem: str):
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None

def split_dataset(items, val_split, test_split, seed):
    random.Random(seed).shuffle(items)
    n = len(items)
    n_val = int(n * val_split)
    n_test = int(n * test_split)
    val = items[:n_val]
    test = items[n_val:n_val + n_test]
    train = items[n_val + n_test:]
    return {"train": train, "val": val, "test": test}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--out-dir", type=str, default="data/processed")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    images_dir = raw_dir / "IMAGES"
    annots_dir = raw_dir / "ANNOTATIONS"
    out_dir = Path(args.out_dir)

    if not images_dir.exists() or not annots_dir.exists():
        raise FileNotFoundError(
            f"Expected {images_dir} and {annots_dir} to exist. "
        )
    xml_files = sorted(annots_dir.glob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No .xml annotation files found in {annots_dir}")
    print(f"Found {len(xml_files)} annotation files. Parsing...")

    valid_items = []
    skipped = 0
    for xml_path in tqdm(xml_files, desc="Parsing VOC annotations"):
        filename, yolo_lines = parse_voc_annotation(xml_path)
        stem = Path(filename).stem if filename else xml_path.stem
        img_path = find_image_for_annotation(images_dir, stem)
        if img_path is None or not yolo_lines:
            skipped += 1
            continue
        valid_items.append((img_path, yolo_lines))

        print(f"Valid image/annotation pairs: {len(valid_items)} (skipped: {skipped})")
        
    splits = split_dataset(valid_items, args.val_split, args.test_split, args.seed)
    for split_name, items in splits.items():
        img_out = out_dir / "images" / split_name
        lbl_out = out_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img_path, yolo_lines in tqdm(items, desc=f"Writing {split_name}"):
            shutil.copy2(img_path, img_out / img_path.name)
            label_path = lbl_out / f"{img_path.stem}.txt"
            label_path.write_text("\n".join(yolo_lines) + "\n")
        print(f"{split_name}: {len(items)} images")
   
    data_yaml = {
        "path": str(out_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASSES),
        "names": CLASSES,
    }
    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False)
        print(f"\nDone. data.yaml written to {yaml_path}")
        
if __name__ == "__main__":
    main()








 













