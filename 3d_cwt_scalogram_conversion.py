
import os
import cv2
import numpy as np
import torch
from PIL import Image
import pywt
from ultralytics import YOLO
import re
from collections import defaultdict, deque

# === Configuration ===
IMAGE_DIR = r"C:\Users\addmin\Downloads\cwt_scalogram\sorted_dataset\fire"
SCALOGRAM_DIR = r"C:\Users\addmin\Downloads\cwt_scalogram\scalogram_kaggel_compression"
YOLO_MODEL_PATH = "best_eo.pt"
TARGET_CLASS_IDS = [0, 1, 2]
SCALE_RANGE = np.arange(1, 64)
SCALOGRAM_SIZE = (227, 227)
TEMPORAL_WINDOW = 3
ALLOWED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

os.makedirs(SCALOGRAM_DIR, exist_ok=True)

print("🚀 Loading YOLO model...")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
yolo = YOLO(YOLO_MODEL_PATH).to(device)
print("📋 YOLO classes:", yolo.names)

def sanitize_filename(filename):
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    filename = re.sub(r'\s+', '_', filename)
    return filename[:100]

def compute_3d_cwt(roi_sequence, scales=SCALE_RANGE, wavelet="morl"):
    try:
        coeffs = []
        for roi in roi_sequence:
            roi_resized = cv2.resize(roi, (64, 64))
            gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
            signal = np.mean(gray, axis=1)
            coeff, _ = pywt.cwt(signal, scales, wavelet)
            coeffs.append(np.abs(coeff))
        return np.stack(coeffs, axis=0)
    except Exception as e:
        print(f"❌ 3D CWT error: {e}")
        return None

def save_3d_scalogram(scalogram_3d, class_id, base_name, frame_idx):
    if scalogram_3d is None or np.max(scalogram_3d) == 0:
        print(f"[⚠️] Skipping empty 3D scalogram for class {class_id}")
        return

    combined = np.max(scalogram_3d, axis=0)
    combined_resized = cv2.resize(combined, SCALOGRAM_SIZE)
    normalized = (combined_resized / np.max(combined_resized) * 255).astype(np.uint8)
    color_mapped = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

    safe_name = sanitize_filename(base_name)
    filename = f"{safe_name}_frame{frame_idx:04d}.png"
    class_dir = os.path.join(SCALOGRAM_DIR, f"class_{class_id}")
    os.makedirs(class_dir, exist_ok=True)

    full_path = os.path.join(class_dir, filename)
    if len(full_path) > 255:
        print(f"[⚠️] Truncating path: {full_path}")
        filename = filename[:100]
        full_path = os.path.join(class_dir, filename)

    print(f"[🧾 Saving to]: {full_path}")
    try:
        with open(full_path, "wb") as f:
            Image.fromarray(cv2.cvtColor(color_mapped, cv2.COLOR_BGR2RGB)).save(f, format="WEBP", optimize=True,
                                                                                compress_level=9)
        print(f"[💾] Saved 3D CWT scalogram for class {class_id}: {filename}")
    except Exception as e:
        print(f"❌ Failed to save image: {e}")

def process_images(image_dir):
    print("🖼️ Processing images from folder:", image_dir)
    filenames = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(ALLOWED_EXTENSIONS))
    roi_queues = defaultdict(lambda: deque(maxlen=TEMPORAL_WINDOW))

    for idx, filename in enumerate(filenames):
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue

        results = yolo.predict(source=image, save=False, verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            continue

        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for i, (cls, conf) in enumerate(zip(classes, confidences)):
            x1, y1, x2, y2 = map(int, boxes[i])
            int_cls = int(cls)
            if int_cls in TARGET_CLASS_IDS and conf > 0.3 and x2 > x1 and y2 > y1:
                roi = image[y1:y2, x1:x2]
                key = str(int_cls)
                roi_queues[key].append(roi)
                print(f"[🌀 Image {idx}] Class {int_cls} ROI added, queue len = {len(roi_queues[key])}")

                if len(roi_queues[key]) == TEMPORAL_WINDOW:
                    scalogram = compute_3d_cwt(list(roi_queues[key]))
                    save_3d_scalogram(scalogram, int_cls, filename.split('.')[0], idx)

if __name__ == "__main__":
    process_images(IMAGE_DIR)
