"""
==============================================================================
AI-Based Vision Screening of Refractive Errors
Dataset Initializer & Synthetic Sample Generator (Spectacle Need Prediction)
==============================================================================
Creates folders:
- dataset/train/no_spectacles (Normal Vision / No Spectacles Required)
- dataset/train/spectacles_needed (Refractive Error / Spectacles Required)
- dataset/val/no_spectacles
- dataset/val/spectacles_needed

Populates starter samples so training and API inference can run immediately.
You can replace or augment these folders with your real dataset at any time.
==============================================================================
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

CLASSES = ["no_spectacles", "spectacles_needed"]
SPLITS = {"train": 40, "val": 15}
IMG_SIZE = (224, 224)

def generate_synthetic_eye(class_name, index):
    """
    Generates ocular image variations:
    - no_spectacles: Symmetrical iris & pupil, crisp single-point corneal reflection.
    - spectacles_needed: Optical blur, asymmetric corneal reflections, or toric distortion.
    """
    # Random skin tone backdrop
    base_color = (
        random.randint(210, 240),
        random.randint(170, 205),
        random.randint(150, 185)
    )
    img = Image.new("RGB", IMG_SIZE, base_color)
    draw = ImageDraw.Draw(img)

    # 1. Sclera
    cx, cy = 112, 112
    sclera_rx = random.randint(85, 95)
    sclera_ry = random.randint(52, 60)
    draw.ellipse([cx - sclera_rx, cy - sclera_ry, cx + sclera_rx, cy + sclera_ry],
                 fill=(246, 244, 241), outline=(200, 160, 150), width=2)

    # 2. Iris
    iris_rx = random.randint(38, 44)
    iris_ry = iris_rx

    if class_name == "spectacles_needed":
        # Introduce mild asymmetry or eccentricity
        if random.random() > 0.5:
            iris_rx = int(iris_rx * 1.15)
            iris_ry = int(iris_ry * 0.88)

    iris_colors = [
        (85, 52, 28),    # Brown
        (115, 78, 42),   # Amber
        (62, 102, 120),  # Blue
        (68, 112, 78)    # Green
    ]
    draw.ellipse([cx - iris_rx, cy - iris_ry, cx + iris_rx, cy + iris_ry],
                 fill=random.choice(iris_colors), outline=(25, 18, 10), width=2)

    # 3. Pupil
    pupil_r = random.randint(14, 19)
    draw.ellipse([cx - pupil_r, cy - pupil_r, cx + pupil_r, cy + pupil_r], fill=(12, 12, 12))

    # 4. Corneal Reflection Highlights
    if class_name == "spectacles_needed":
        # Multiple / distorted reflection points indicative of astigmatism or refractive shift
        draw.ellipse([cx - 9, cy - 9, cx - 3, cy - 3], fill=(255, 255, 255))
        if random.random() > 0.4:
            draw.ellipse([cx + 5, cy + 3, cx + 9, cy + 7], fill=(240, 240, 255))
    else:
        # Sharp single-point reflection
        draw.ellipse([cx - 7, cy - 7, cx - 2, cy - 2], fill=(255, 255, 255))

    # 5. Image filtering
    if class_name == "spectacles_needed" and random.random() > 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=0.9))
    else:
        img = img.filter(ImageFilter.SMOOTH_MORE)

    # Add realistic sensor noise
    arr = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, 3.5, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def setup_dataset():
    """Initializes dataset folders and populates sample images."""
    print(f"Creating dataset structure in: {DATASET_DIR}")
    
    for split, count in SPLITS.items():
        for cls in CLASSES:
            folder = os.path.join(DATASET_DIR, split, cls)
            os.makedirs(folder, exist_ok=True)
            
            existing = len(os.listdir(folder))
            if existing < count:
                print(f"  Generating {count - existing} samples for {split}/{cls}...")
                for i in range(existing, count):
                    img = generate_synthetic_eye(cls, i)
                    img_path = os.path.join(folder, f"sample_{cls}_{i+1:03d}.jpg")
                    img.save(img_path, quality=95)

    print("\nDataset successfully initialized!")
    for split in SPLITS.keys():
        for cls in CLASSES:
            p = os.path.join(DATASET_DIR, split, cls)
            print(f" - {p} ({len(os.listdir(p))} images)")
    print("\nDrop your real dataset images into these folders whenever you are ready!")

if __name__ == "__main__":
    setup_dataset()
