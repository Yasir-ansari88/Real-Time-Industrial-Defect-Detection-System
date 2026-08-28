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

    print("Press 'q' to quit .")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video stream (or camera disconnected).")
                break

            t0 = time.perf_counter()
            results = model.predict(frame, conf=args.conf, verbose=False)[0]
            annotated = results.plot() 
            t1 = time.perf_counter()

            instant_fps = 1.0 / max(t1 - t0, 1e-6)
            smoothed_fps = instant_fps if frame_count == 0 else 0.9 * smoothed_fps + 0.1 * instant_fps
            cv2.putText(
                annotated, f"FPS: {smoothed_fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2,
            )

            if writer is not None:
                writer.write(annotated)

            if not args.no_display:
                cv2.imshow("Defect Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Quit requested.")
                    break

            frame_count += 1

    except KeyboardInterrupt:
        print("\nInterrupted.")

    total_elapsed = time.perf_counter() - t_start
    cap.release()
    if writer is not None:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    if frame_count > 0:
        print(f"\nProcessed {frame_count} frames in {total_elapsed:.1f}s "
              f"(avg {frame_count / total_elapsed:.1f} FPS overall).")


if __name__ == "__main__":
    main()
