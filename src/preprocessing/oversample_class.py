import argparse
from pathlib import Path

import albumentations as A
import cv2
from tqdm import tqdm

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
MIN_VISIBILITY = 0.3


def build_transform():
    return A.Compose(
        [
            A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT_101, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.OneOf(
                [
                    A.GaussNoise(var_limit=(10.0, 40.0), p=1.0),
                    A.ISONoise(p=1.0),
                ],
                p=0.4,
            ),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.RandomGamma(gamma_limit=(75, 125), p=0.4),
            
            A.OneOf(
                [
                    A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=1.0),
                    A.Blur(blur_limit=3, p=1.0),
                ],
                p=0.3,
            ),
        ],
        bbox_params=A.BboxParams(
            format="yolo", label_fields=["class_labels"], min_visibility=MIN_VISIBILITY
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
    parser.add_argument("--target-classes", type=str, nargs="+", default=["crazing"])
    parser.add_argument("--multiplier", type=int, default=3,
                         help="Extra augmented copies per original image containing a target class")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for cls in args.target_classes:
        if cls not in CLASS_TO_IDX:
            raise ValueError(f"Unknown class '{cls}'. Valid options: {CLASSES}")
    target_idxs = {CLASS_TO_IDX[c] for c in args.target_classes}

    data_dir = Path(args.data_dir)
    train_img_dir = data_dir / "images" / "train"
    train_lbl_dir = data_dir / "labels" / "train"

    if not train_img_dir.exists():
        raise FileNotFoundError(f"{train_img_dir} not found.")

    original_images = sorted(
        p for p in train_img_dir.glob("*.jpg")
        if "_aug" not in p.stem and "_oversample" not in p.stem
    )

    target_images = []
    for img_path in original_images:
        label_path = train_lbl_dir / f"{img_path.stem}.txt"
        _, class_labels = read_yolo_labels(label_path)
        if any(c in target_idxs for c in class_labels):
            target_images.append(img_path)

    print(f"Found {len(target_images)} original images containing "
          f"{args.target_classes} (out of {len(original_images)} total originals).")
    print(f"Generating {args.multiplier} extra augmented copies each "
          f"(~{len(target_images) * args.multiplier} new images)...")

    transform = build_transform()
    created, dropped = 0, 0

    for img_path in tqdm(target_images, desc="Oversampling"):
        label_path = train_lbl_dir / f"{img_path.stem}.txt"
        bboxes, class_labels = read_yolo_labels(label_path)
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        for i in range(args.multiplier):
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_bboxes = augmented["bboxes"]
            aug_labels = augmented["class_labels"]

            if not aug_bboxes:
                dropped += 1
                continue

            out_stem = f"{img_path.stem}_oversample{i + 1}"
            cv2.imwrite(str(train_img_dir / f"{out_stem}.jpg"), augmented["image"])
            write_yolo_labels(train_lbl_dir / f"{out_stem}.txt", aug_bboxes, aug_labels)
            created += 1

    print(f"\nDone. Created {created} oversampled image/label pairs.")
    if dropped:
        print(f"Skipped {dropped} attempts (all target boxes fell outside frame).")

    total_train = len(list(train_img_dir.glob("*.jpg")))
    print(f"Train set size now: {total_train} images.")
    print("\nRe-run training ")


if __name__ == "__main__":
    main()
