import os
from PIL import Image
import torch
from model import SpectacleVisionNet, predict_image_pipeline

device = torch.device("cpu")
model = SpectacleVisionNet(num_spectacle_classes=2, num_conditions=4)
model.load_state_dict(torch.load("saved_models/spectacle_screening_model.pth", map_location=device))
model.eval()

test_samples = [
    ("Normal Test Image", r"D:\Major Project\MajorProject\dataset\AI generated Dataset\normal"),
    ("Myopia Error Image", r"D:\Major Project\MajorProject\dataset\AI generated Dataset\myopia"),
    ("Hyperopia Error Image", r"D:\Major Project\MajorProject\dataset\AI generated Dataset\hyperopia"),
    ("Astigmatism Error Image", r"D:\Major Project\MajorProject\dataset\AI generated Dataset\astigmatism"),
]

for label, folder in test_samples:
    if os.path.exists(folder):
        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))]
        if files:
            img_path = os.path.join(folder, files[0])
            img = Image.open(img_path)
            res = predict_image_pipeline(model, img, device)
            print("=" * 50)
            print(f"Sample: {label}")
            print(f"Image File: {files[0]}")
            print(f"Result: {res['status']} (Spectacles Needed: {res['spectacles_needed']})")
            print(f"Condition: {res['detected_condition']}")
            print(f"Confidence: {res['confidence_percentage']}%")
            print(f"Probabilities: {res['probabilities']}")
            print(f"Advice: {res['clinical_advice']}")
