"""
==============================================================================
AI Vision Screening: Spectacle Requirement Determination Model (PyTorch)
==============================================================================
Predicts whether an individual requires spectacles based on eye image screening:
- Normal Eye Image -> "No Spectacles Needed" (spectacles_needed = False)
- Error / Defect Eye Image (Myopia, Hyperopia, Astigmatism) -> "Spectacles Required" (spectacles_needed = True)

Architecture:
- High-efficiency Custom Mobile Deep CNN (SpectacleVisionNet)
- Dual Head: Spectacle Need Classification & Error Condition Diagnostics
==============================================================================
"""

import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Class Definitions
SPECTACLE_CLASSES = ["No Spectacles Needed", "Spectacles Required"]
CONDITION_CLASSES = ["normal", "myopia", "hyperopia", "astigmatism"]

IMAGE_SIZE = (128, 128)

# Preprocessing transforms
def get_inference_transforms():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def get_train_transforms():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


class SpectacleVisionNet(nn.Module):
    """
    High-efficiency Deep CNN for Spectacle Requirement Screening.
    Dual output heads:
    1. Binary Spectacle Decision (No Spectacles Needed vs Spectacles Required)
    2. Detailed Condition Diagnosis (Normal, Myopia, Hyperopia, Astigmatism)
    """
    def __init__(self, num_spectacle_classes=2, num_conditions=4):
        super(SpectacleVisionNet, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 64x64

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 32x32

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # 16x16

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)) # 1x1
        )

        # Shared representation layer
        self.shared_fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        # Head 1: Binary Spectacle Requirement (0 = No Spectacles Needed, 1 = Spectacles Required)
        self.spectacle_head = nn.Linear(128, num_spectacle_classes)

        # Head 2: Condition Classification (0 = normal, 1 = myopia, 2 = hyperopia, 3 = astigmatism)
        self.condition_head = nn.Linear(128, num_conditions)

    def forward(self, x):
        feat = self.features(x)
        feat = torch.flatten(feat, 1)
        shared = self.shared_fc(feat)
        
        spectacle_logits = self.spectacle_head(shared)
        condition_logits = self.condition_head(shared)
        
        return spectacle_logits, condition_logits


def predict_image_pipeline(model, image: Image.Image, device="cpu", symptom: str = None):
    """
    Runs complete screening prediction pipeline on a PIL image.
    Pure deep learning model inference trained on the 3 datasets:
    - Normal eye image -> "No Spectacles Needed" (spectacles_needed = False)
    - Error eye image (myopia, hyperopia, astigmatism) -> "Spectacles Required" (spectacles_needed = True)
    """
    model.eval()
    transform = get_inference_transforms()
    
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        spectacle_logits, condition_logits = model(tensor)
        spectacle_probs = torch.softmax(spectacle_logits, dim=1).cpu().numpy()[0]
        condition_probs = torch.softmax(condition_logits, dim=1).cpu().numpy()[0]

    prob_normal = float(spectacle_probs[0])
    prob_error = float(spectacle_probs[1])

    # Condition index predicted by CNN
    cond_idx = int(np.argmax(condition_probs))
    cond_name = CONDITION_CLASSES[cond_idx]

    # Spectacles needed determination directly from trained model
    spectacles_needed = prob_error > prob_normal

    condition_display_map = {
        "normal": "Normal Ocular Vision (Emmetropic)",
        "myopia": "Myopia (Nearsightedness)",
        "hyperopia": "Hyperopia (Farsightedness)",
        "astigmatism": "Astigmatism (Corneal Asymmetry)"
    }

    if spectacles_needed:
        status_label = "Spectacles Required"
        clinical_advice = (
            f"Screening analysis indicates ocular refractive anomaly characteristic of {condition_display_map.get(cond_name, 'Refractive Error')}. "
            f"Corrective prescription spectacles are recommended. Please schedule an examination with an optometrist."
        )
        badge_color = "warning"
        confidence = prob_error
    else:
        status_label = "No Spectacles Needed"
        cond_name = "normal"
        clinical_advice = (
            "Ocular alignment, corneal reflection clarity, and focal characteristics appear within normal healthy parameters. "
            "No corrective spectacles are indicated at this time."
        )
        badge_color = "success"
        confidence = prob_normal
        clinical_advice = (
            "Ocular alignment, corneal reflection clarity, and focal characteristics appear within normal healthy parameters. "
            "No corrective spectacles are indicated at this time."
        )
        badge_color = "success"
        confidence = prob_normal

    condition_prob_dict = {
        "Normal Vision": round(float(condition_probs[0]) * 100, 2),
        "Myopia": round(float(condition_probs[1]) * 100, 2),
        "Hyperopia": round(float(condition_probs[2]) * 100, 2),
        "Astigmatism": round(float(condition_probs[3]) * 100, 2),
    }

    return {
        "success": True,
        "spectacles_needed": spectacles_needed,
        "status": status_label,
        "detected_condition": condition_display_map.get(cond_name, cond_name.capitalize()),
        "confidence_percentage": round(confidence * 100, 2),
        "clinical_advice": clinical_advice,
        "badge_color": badge_color,
        "probabilities": {
            "No Spectacles Needed": round(prob_normal * 100, 2),
            "Spectacles Required": round(prob_error * 100, 2)
        },
        "condition_breakdown": condition_prob_dict
    }
