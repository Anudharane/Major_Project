"""
==============================================================================
AI Vision Screening: Binary Spectacle Requirement Training Pipeline
==============================================================================
Trains a ResNet-18 Transfer Learning model with optical feature fusion
strictly on the user's binary dataset:
- "Spectacles not required" (Class 0: No Spectacles Needed)
- "spectacles needed" (Class 1: Spectacles Needed)

Predicts ONLY:
- User Needs Spectacles (True) vs No Spectacles Needed (False)
==============================================================================
"""

import os
import sys
import json
import time
import argparse
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from model import (
    ResNet18SpectacleBinaryNet,
    get_train_transforms,
    get_inference_transforms,
    SPECTACLE_CLASSES
)
from image_refiner import extract_optical_feature_tensor, refine_ocular_image

# Paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent.parent
SAVED_MODELS_DIR = BACKEND_DIR / "saved_models"
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = SAVED_MODELS_DIR / "spectacle_screening_model.pth"
METADATA_SAVE_PATH = SAVED_MODELS_DIR / "model_metadata.json"


class SpectacleBinaryDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None, refine=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.refine = refine

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["absolute_path"]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (128, 128), (0, 0, 0))

        if self.refine:
            image, _ = refine_ocular_image(image)

        opt_feats = extract_optical_feature_tensor(image)
        opt_tensor = torch.tensor(opt_feats, dtype=torch.float)

        if self.transform:
            img_tensor = self.transform(image)
        else:
            img_tensor = transforms.ToTensor()(image)

        label = int(row["spectacles_needed"])
        return img_tensor, opt_tensor, label


def load_binary_dataset(samples_per_class: int = 3000) -> pd.DataFrame:
    csv_path = BACKEND_DIR / "dataset" / "csv" / "spectacles_binary_dataset.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Binary dataset CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Loaded binary dataset containing {len(df):,} total records.", flush=True)

    # Class 0: Not Required
    df_class0 = df[df["spectacles_needed"] == 0]
    n0 = min(len(df_class0), samples_per_class)
    sampled_c0 = df_class0.sample(n=n0, random_state=42)

    # Class 1: Needed (sample evenly across all subcategories)
    df_class1 = df[df["spectacles_needed"] == 1]
    cats = df_class1["category"].unique()
    per_cat = max(1, samples_per_class // len(cats))
    sampled_c1_list = []
    for cat in cats:
        cat_df = df_class1[df_class1["category"] == cat]
        n_c = min(len(cat_df), per_cat)
        sampled_c1_list.append(cat_df.sample(n=n_c, random_state=42))
    sampled_c1 = pd.concat(sampled_c1_list)

    balanced_df = pd.concat([sampled_c0, sampled_c1]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"\nBalanced Training Dataset Size: {len(balanced_df):,} images", flush=True)
    print(balanced_df.groupby(["label", "category"]).size(), flush=True)
    return balanced_df


def train_binary_model(epochs=4, batch_size=32, lr=1e-3, samples_per_class=3000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================", flush=True)
    print(f"Training ResNet-18 Optical Feature-Fused Binary Spectacle Model", flush=True)
    print(f"Device: {device} | Epochs: {epochs} | Batch Size: {batch_size}", flush=True)
    print(f"Classes: No Spectacles Needed (0) vs Spectacles Needed (1)", flush=True)
    print(f"=======================================================", flush=True)

    df = load_binary_dataset(samples_per_class=samples_per_class)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["spectacles_needed"]
    )
    print(f"Training split: {len(train_df):,} images | Validation split: {len(val_df):,} images", flush=True)

    train_dataset = SpectacleBinaryDataset(train_df, transform=get_train_transforms())
    val_dataset = SpectacleBinaryDataset(val_df, transform=get_inference_transforms())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = ResNet18SpectacleBinaryNet(
        num_classes=2,
        optical_feat_dim=4,
        pretrained=True,
        freeze_early=True
    ).to(device)

    # Differential learning rate
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "layer3" in name or "layer4" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.1},
        {"params": head_params, "lr": lr}
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    history = []
    start_train_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, opt_feats, labels in train_loader:
            images = images.to(device)
            opt_feats = opt_feats.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images, opt_feats)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

        scheduler.step()
        train_loss = running_loss / max(1, total)
        train_acc = (correct / max(1, total)) * 100.0

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        c0_correct = 0
        c0_total = 0
        c1_correct = 0
        c1_total = 0

        with torch.no_grad():
            for images, opt_feats, labels in val_loader:
                images = images.to(device)
                opt_feats = opt_feats.to(device)
                labels = labels.to(device)

                logits = model(images, opt_feats)
                loss = criterion(logits, labels)

                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(logits, dim=1)

                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)

                for p, y in zip(preds, labels):
                    if y.item() == 0:
                        c0_total += 1
                        if p.item() == 0:
                            c0_correct += 1
                    else:
                        c1_total += 1
                        if p.item() == 1:
                            c1_correct += 1

        val_loss = val_loss / max(1, val_total)
        val_acc = (val_correct / max(1, val_total)) * 100.0
        spec_0_acc = (c0_correct / max(1, c0_total)) * 100.0 if c0_total > 0 else 0.0
        spec_1_acc = (c1_correct / max(1, c1_total)) * 100.0 if c1_total > 0 else 0.0
        epoch_sec = round(time.time() - epoch_start, 1)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_sec}s) | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.1f}% | "
              f"Val Acc: {val_acc:.1f}% (No Spectacles: {spec_0_acc:.1f}%, Spectacles Needed: {spec_1_acc:.1f}%)", flush=True)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
            "val_no_spectacles_acc": round(spec_0_acc, 2),
            "val_spectacles_needed_acc": round(spec_1_acc, 2)
        })

        if val_acc > best_val_acc or epoch == 1:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ==> Saved best model checkpoint to {MODEL_SAVE_PATH.name} (Val Acc: {val_acc:.1f}%)", flush=True)

    total_sec = round(time.time() - start_train_time, 1)
    print(f"\nTraining Complete in {total_sec}s! Peak Validation Accuracy: {best_val_acc:.2f}%", flush=True)

    metadata = {
        "model_architecture": "ResNet-18 Feature-Fused Binary Classifier",
        "task": "Spectacle Requirement Screening (Binary)",
        "best_val_accuracy": round(best_val_acc, 2),
        "total_training_time_sec": total_sec,
        "classes": SPECTACLE_CLASSES,
        "history": history
    }
    with open(METADATA_SAVE_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    return model, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Binary Spectacle Screening Model")
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--samples_per_class", type=int, default=2500, help="Samples per class")
    args = parser.parse_args()

    train_binary_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        samples_per_class=args.samples_per_class
    )
