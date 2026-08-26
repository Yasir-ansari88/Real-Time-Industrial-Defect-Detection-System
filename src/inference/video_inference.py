import argparse
import time
import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    parser.add_argument("--source", type=str, default="0", help="Webcam index or path to a video file")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--save-path", type=str, default=None, help="If set, saves the annotated video here")
    parser.add_argument("--no-display", action="store_true", help="Don't open a live window")
    args = parser.parse_args()

    model = YOLO(args.weights)
