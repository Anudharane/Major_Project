import os
from pathlib import Path
from PIL import Image
import torch
from model import ResNet18SpectacleBinaryNet, predict_image_pipeline

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
MODEL_PATH = SAVED_MODELS_DIR / "spectacle_screening_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18SpectacleBinaryNet(num_classes=2, optical_feat_dim=4, pretrained=False)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

test_samples = [
    ("Spectacles Not Required (Normal)", DATASET_DIR / "Spectacles not required" / "normal"),
    ("Spectacles Not Required (Normal 1)", DATASET_DIR / "Spectacles not required" / "normal1"),
    ("Spectacles Needed (Astigmatism)", DATASET_DIR / "spectacles needed" / "astigmatism"),
    ("Spectacles Needed (Hyperopia)", DATASET_DIR / "spectacles needed" / "hyperopia"),
    ("Spectacles Needed (Hyperopia 2)", DATASET_DIR / "spectacles needed" / "hyperopia2"),
    ("Spectacles Needed (Myopia)", DATASET_DIR / "spectacles needed" / "myopia"),
    ("Spectacles Needed (Myopia 1)", DATASET_DIR / "spectacles needed" / "myopia1"),
]

print("=" * 65)
print("Testing Binary Spectacle Screening Model on Real Dataset Images")
print("=" * 65)

for label, folder in test_samples:
    if folder.exists():
        files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))]
        if files:
            img_path = folder / files[0]
            img = Image.open(img_path)
            res = predict_image_pipeline(model, img, device)
            print("=" * 65)
            print(f"Dataset Category : {label}")
            print(f"Sample File      : {files[0]}")
            print(f"Prediction       : {res['status']} (Spectacles Needed: {res['spectacles_needed']})")
            print(f"Confidence       : {res['confidence_percentage']}%")
            print(f"Probabilities    : {res['probabilities']}")
            print(f"Extracted Features: {res['extracted_features']}")
            print(f"Guidance Advice  : {res['clinical_advice']}")
