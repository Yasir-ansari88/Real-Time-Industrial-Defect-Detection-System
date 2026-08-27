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
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save_path, fourcc, fps_in, (width, height))
        print(f"Saving annotated output to {args.save_path}")

    frame_count = 0
    t_start = time.perf_counter()
    smoothed_fps = 0.0
    