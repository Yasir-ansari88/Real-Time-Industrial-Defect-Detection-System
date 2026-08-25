import argparse
import time
import os

from ultralytics import YOLO

def benchmark(model_path: str, image_path:str, runs: int, warmup: int=10):
    model = YOLO(model_path)
    device = "cpu" if model_path.endswith(".onnx") else None

    for _ in range(warmup):
        model.predict(image_path, verbose=False, device = device)

    start = time.perf_counter()
    for _ in range(runs):
        model.predict(image_path, verbose=False, device = device)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / runs) * 1000
    fps = runs / elapsed
    return avg_ms, fps

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pt", type=str, help="Path to .pt weights")
    parser.add_argument("--onnx", type=str, help="Path to .onnx weights")
    parser.add_argument("--engine", type=str, help="Path to .engine TensorAT weights")
    parser.add_argument("--image", type=str, required=True, help="Sample images to run")
    parser.add_argument("--runs", type=int, default=100)
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"{args.image} not found.")

    candidates = [
        ("Pytorch (.pt)", args.pt),
        ("Onnx (.onnex)", args.onnex),
        ("TensoRT (.engine)", args.engine)
    ]

    results = []
    for label, path in candidates:
        if not path:
            continue
        if not os.path.exists(path):
            print(f"Skipping {label} : {path} not found.")
            continue
        print(f"\nBenchmarking {label} ({args.runs} runs, {10} warmup)...")
        avg_ms, fps = benchmark(path, args.image, args.runs)
        results.append((label, avg_ms, fps))
        print(f" {label}: {avg_ms:.2f} ms/image, {fps:.2f} FPS")

    if not results:
        print("No valid model paths given.")
        return

    print("SUMMARY")
    baseline_ms = results[0][1]
    for label, avg_ms, fps in results:
        speedup = baseline_ms / avg_ms
        print(f"{label:22s} {avg_ms:7.2f} ms {fps:8.1f} FPS ({speedup:.2f}x vs {results[0][0]})")

if __name__ == "__main__":
    main()