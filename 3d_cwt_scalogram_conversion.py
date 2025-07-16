import os
import cv2
import numpy as np
import torch
from PIL import Image
import pywt
from collections import deque

# === Configuration ===
IMAGE_DIR = r"C:\Users\addmin\Downloads\cwt_scalogram\youtube_scalogram\nonmfcc_dataset\nonfire"
SCALOGRAM_DIR = r"C:\Users\addmin\Downloads\cwt_scalogram\youtube_scalogram\nonfire"
SCALE_RANGE = np.arange(1, 64)
SCALOGRAM_SIZE = (227, 227)
TEMPORAL_WINDOW = 3
ALLOWED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')

os.makedirs(SCALOGRAM_DIR, exist_ok=True)

def sanitize_filename(filename):
    return "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in filename)[:100]

def compute_3d_cwt(image_sequence, scales=SCALE_RANGE, wavelet="morl"):
    try:
        coeffs = []
        for image in image_sequence:
            image_resized = cv2.resize(image, (64, 64))
            gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
            signal = np.mean(gray, axis=1)
            coeff, _ = pywt.cwt(signal, scales, wavelet)
            coeffs.append(np.abs(coeff))
        return np.stack(coeffs, axis=0)
    except Exception as e:
        print(f"❌ 3D CWT error: {e}")
        return None

def save_3d_scalogram(scalogram_3d, base_name, frame_idx):
    if scalogram_3d is None or np.max(scalogram_3d) == 0:
        print(f"[⚠️] Skipping empty 3D scalogram")
        return

    combined = np.max(scalogram_3d, axis=0)
    combined_resized = cv2.resize(combined, SCALOGRAM_SIZE)
    normalized = (combined_resized / np.max(combined_resized) * 255).astype(np.uint8)
    color_mapped = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

    safe_name = sanitize_filename(base_name)
    filename = f"{safe_name}_frame{frame_idx:04d}.png"
    full_path = os.path.join(SCALOGRAM_DIR, filename)

    try:
        Image.fromarray(cv2.cvtColor(color_mapped, cv2.COLOR_BGR2RGB)).save(
            full_path, format="PNG", optimize=True
        )
        print(f"[💾] Saved scalogram: {filename}")
    except Exception as e:
        print(f"❌ Failed to save image: {e}")

def process_image_sequence(image_dir):
    print("🖼️ Processing raw images...")
    filenames = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(ALLOWED_EXTENSIONS))
    image_buffer = deque(maxlen=TEMPORAL_WINDOW)

    for idx, filename in enumerate(filenames):
        image_path = os.path.join(image_dir, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue

        image_buffer.append(image)

        if len(image_buffer) == TEMPORAL_WINDOW:
            scalogram = compute_3d_cwt(list(image_buffer))
            save_3d_scalogram(scalogram, filename.split('.')[0], idx)

if __name__ == "__main__":
    process_image_sequence(IMAGE_DIR)
