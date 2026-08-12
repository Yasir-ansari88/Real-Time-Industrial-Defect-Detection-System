import argparse
import random
from pathlib import Path

import albumentations as A
import cv2
from tqdm import tqdm

MIN_VISIBILITY = 0.3

def build_transform():
    return A.Compose(
        [
            A.Rotate(limit=25, border_mode=cv2.BORDER_REFLECT_101, p=0.6),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.OneOf(
                [
                    A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                    A.ISONoise(p=1.0),
                ],
                p=0.4,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.25, contrast_limit=0.25, p=0.6
            ),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.Blur(blur_limit=3, p=0.15),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_labels"],
            min_visibility=MIN_VISIBILITY,
        ),
    )

def read_yolo_labels(label_path: Path):
    bboxes, class_labels = [], []
    if not label_path.exists():
        return bboxes, class_labels
    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        cls = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        bboxes.append([cx, cy, w, h])
        class_labels.append(cls)
    return bboxes, class_labels

def write_yolo_labels(label_path: Path, bboxes, class_labels):
    lines = [
        f"{cls} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}"
        for cls, bb in zip(class_labels, bboxes)
    ]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--num-augmentations", type = int, default=3, help="How many augmented copies to generate per original train image")
    parser.add_argument("--seed", type= int, default = 42)
    args = parser.parse_args()

    random.seed(args.seed)

    data_dir = Path(args.data_dir)
    train_img_dir = data_dir / "images" / "train"
    train_lbl_dir = data_dir / "labels" / "train"

    if not train_img_dir.exists():
        raise FileNotFoundError(
            f"{train_img_dir} not found."
        )

    original_images = sorted(
        p for p in train_img_dir.glob("*.jpg") if "_aug" not in p.stem
    )
    if not original_images:
        raise FileNotFoundError(f"No original .jpg images found in {train_img_dir}")

    print(f"Found {len(original_images)} original training images.")
    print(f"Generating {args.num_augmentations} augmented copies each " 
          f"(~{len(original_images) * args.num_augmentations} new images)...")
    