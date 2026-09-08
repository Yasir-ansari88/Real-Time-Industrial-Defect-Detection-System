import requests
import time
import os
import glob
import random

API_URL = "http://localhost:8000/predict"      
DATA_FOLDER = "E:/Real-Time Industrial Defect Detection System/data/processed/images/val"                             # folder containing test images
IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png")
REQUEST_INTERVAL_SECONDS = 2                     
TOTAL_REQUESTS = 100             

def find_images(folder):
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
    return images


def send_request(image_path):
    try:
        with open(image_path, "rb") as f:
            files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
            start = time.time()
            response = requests.post(API_URL, files=files, timeout=15)
            elapsed = time.time() - start

        if response.status_code == 200:
            result = response.json()
            num_detections = len(result.get("detections", result.get("predictions", [])))
            print(f"✅ {os.path.basename(image_path):30s} | "
                  f"Status: {response.status_code} | "
                  f"Time: {elapsed:.3f}s | "
                  f"Detections: {num_detections}")
        else:
            print(f"⚠️  {os.path.basename(image_path):30s} | "
                  f"Status: {response.status_code} | "
                  f"Response: {response.text[:100]}")

    except requests.exceptions.RequestException as e:
        print(f"❌ Error sending {image_path}: {e}")


def main():
    images = find_images(DATA_FOLDER)

    if not images:
        print(f"No images found in '{DATA_FOLDER}'.")
        return

    print(f"Found {len(images)} image(s) in '{DATA_FOLDER}'.")
    print(f"Sending requests to {API_URL} every {REQUEST_INTERVAL_SECONDS}s...")

    count = 0
    try:
        while TOTAL_REQUESTS is None or count < TOTAL_REQUESTS:
            image_path = random.choice(images)
            send_request(image_path)
            count += 1
            time.sleep(REQUEST_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print(f"\nStopped after {count} requests.")


if __name__ == "__main__":
    main()