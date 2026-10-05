"""
==============================================================================
AI Vision Screening: Binary Spectacle Requirement Model (PyTorch)
==============================================================================
Binary Classifier: Determines whether a user requires spectacles or not.
Classes:
  - 0: "No Spectacles Needed"
  - 1: "Spectacles Needed"

Architecture:
- ResNet-18 Transfer Learning Backbone (Pretrained on ImageNet)
- Multi-modal Ocular Feature Extraction & Optical Feature Fusion
- Ocular ROI Localization, Illumination Normalization & CLAHE
==============================================================================
"""

import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from image_refiner import refine_ocular_image, extract_optical_feature_tensor, pil_to_base64_thumbnail

SPECTACLE_CLASSES = ["No Spectacles Needed", "Spectacles Needed"]
IMAGE_SIZE = (128, 128)

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
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


class ResNet18SpectacleBinaryNet(nn.Module):
    """
    ResNet-18 Transfer Learning Binary Classifier for Spectacle Requirement.
    Fuses deep visual representations with handcrafted ocular optical features.
    """
    def __init__(self, num_classes=2, optical_feat_dim=4, pretrained=True, freeze_early=True):
        super(ResNet18SpectacleBinaryNet, self).__init__()
        
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
            base = models.resnet18(weights=weights)
        else:
            base = models.resnet18(weights=None)

        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool

        if freeze_early and pretrained:
            for p in [self.conv1, self.bn1, self.layer1, self.layer2]:
                for param in p.parameters():
                    param.requires_grad = False

        # Optical feature projection
        self.optical_fc = nn.Sequential(
            nn.Linear(optical_feat_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )

        # Fused classifier (512 visual + 32 optical = 544 dimensions)
        self.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(512 + 32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, opt_feats=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        feat_img = torch.flatten(x, 1)  # (B, 512)

        if opt_feats is None:
            opt_feats = torch.zeros((feat_img.size(0), 4), device=feat_img.device)
        
        feat_opt = self.optical_fc(opt_feats)
        fused = torch.cat([feat_img, feat_opt], dim=1)  # (B, 544)
        logits = self.classifier(fused)

        return logits


# Aliases for backward compatibility
ResNet18SpectacleNet = ResNet18SpectacleBinaryNet
SpectacleVisionNet = ResNet18SpectacleBinaryNet


def predict_image_pipeline(model, raw_image: Image.Image, device="cpu", symptom: str = None):
    """
    Runs complete binary vision screening pipeline:
    1. Refines ocular ROI, applies CLAHE contrast enhancement & sharpening
    2. Extracts quantitative optical feature vector
    3. Runs ResNet-18 Transfer Learning model
    4. Outputs exclusively whether spectacles are needed or not
    """
    # 1. Refine image & extract optical features
    refined_image, extracted_features = refine_ocular_image(raw_image)
    refined_thumbnail_b64 = pil_to_base64_thumbnail(refined_image)
    opt_vec = extract_optical_feature_tensor(refined_image)

    # 2. Prepare PyTorch tensors
    transform = get_inference_transforms()
    img_tensor = transform(refined_image).unsqueeze(0).to(device)
    opt_tensor = torch.tensor(opt_vec, dtype=torch.float).unsqueeze(0).to(device)

    # 3. Model inference
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor, opt_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    prob_not_needed = float(probs[0])
    prob_needed = float(probs[1])

    # Contextual symptom synergy (if provided)
    if symptom:
        s_lower = symptom.lower()
        if any(w in s_lower for w in ["blur", "reading", "distance", "strain", "headache"]):
            prob_needed = max(prob_needed, 0.70)
            prob_not_needed = 1.0 - prob_needed
        elif "no symptoms" in s_lower or "routine" in s_lower:
            prob_not_needed = max(prob_not_needed, 0.65)
            prob_needed = 1.0 - prob_not_needed

    spectacles_needed = prob_needed > prob_not_needed
    confidence = prob_needed if spectacles_needed else prob_not_needed

    if spectacles_needed:
        status_label = "Spectacles Needed"
        clinical_advice = (
            "Screening analysis indicates ocular refractive error characteristics. "
            "Prescription corrective spectacles are indicated. Please consult a qualified optometrist for an eye examination."
        )
        badge_color = "warning"
    else:
        status_label = "No Spectacles Needed"
        clinical_advice = (
            "Ocular alignment, focal clarity, and corneal reflection appear within normal parameters. "
            "No corrective spectacles are required at this time."
        )
        badge_color = "success"

    return {
        "success": True,
        "model_architecture": "ResNet-18 Feature-Fused Transfer Learning",
        "spectacles_needed": spectacles_needed,
        "status": status_label,
        "confidence_percentage": round(confidence * 100, 2),
        "clinical_advice": clinical_advice,
        "badge_color": badge_color,
        "probabilities": {
            "Spectacles Needed": round(prob_needed * 100, 2),
            "No Spectacles Needed": round(prob_not_needed * 100, 2)
        },
        "extracted_features": extracted_features,
        "refined_image_preview": refined_thumbnail_b64
    }
