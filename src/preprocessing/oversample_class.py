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