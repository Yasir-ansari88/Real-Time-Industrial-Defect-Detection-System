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