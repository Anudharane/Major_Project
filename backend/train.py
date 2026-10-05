"""
==============================================================================
AI Vision Screening: Model Training Pipeline (Multi-Source Balanced)
==============================================================================
Ensures balanced sampling across all 3 datasets and all conditions:
- AI Generated Dataset (normal, myopia, hyperopia, astigmatism)
- Real Time Dataset (normal, myopia)
- Eye Dataset (normal, myopia, hyperopia)

- Normal Vision -> "No Spectacles Needed" (spectacles_needed = 0)
- Refractive Errors (Myopia, Hyperopia, Astigmatism) -> "Spectacles Required" (spectacles_needed = 1)
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
    SpectacleVisionNet,
    get_train_transforms,
    get_inference_transforms,
    SPECTACLE_CLASSES,
    CONDITION_CLASSES
)

# Paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent.parent
SAVED_MODELS_DIR = BACKEND_DIR / "saved_models"
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = SAVED_MODELS_DIR / "spectacle_screening_model.pth"
METADATA_SAVE_PATH = SAVED_MODELS_DIR / "model_metadata.json"

CONDITION_TO_IDX = {name: i for i, name in enumerate(CONDITION_CLASSES)}


class EyeImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["absolute_path"]
        
        if not os.path.exists(img_path):
            rel_path = row.get("relative_path", "")
            img_path = str(PROJECT_ROOT / rel_path)

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (128, 128), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        spectacle_label = int(row["spectacles_needed"])
        condition_str = str(row["condition"]).lower()
        condition_label = CONDITION_TO_IDX.get(condition_str, 0)

        return image, spectacle_label, condition_label


def load_multi_source_balanced_data(samples_per_source_condition: int = 300) -> pd.DataFrame:
    master_csv = BACKEND_DIR / "dataset" / "csv" / "master_dataset.csv"
    if not master_csv.exists():
        print("master_dataset.csv not found. Running CSV generator...", flush=True)
        import create_dataset_csvs
        create_dataset_csvs.generate_csvs()

    df = pd.read_csv(master_csv)
    print(f"Loaded master dataset containing {len(df):,} records.", flush=True)

    # Group by (dataset_source, condition) and sample equally
    groups = df.groupby(["dataset_source", "condition"])
    sampled_frames = []

    for (source, cond), group_df in groups:
        n_sample = min(len(group_df), samples_per_source_condition)
        sampled = group_df.sample(n=n_sample, random_state=42)
        sampled_frames.append(sampled)
        print(f"  - [{source}] {cond}: sampled {len(sampled)} images (out of {len(group_df):,})")

    balanced_df = pd.concat(sampled_frames).sample(frac=1.0, random_state=42).reset_index(drop=True)
    print(f"\nFinal Multi-Source Dataset: {len(balanced_df):,} images", flush=True)
    print(balanced_df.groupby(["label", "condition"]).size(), flush=True)
    return balanced_df


def train_model(epochs=6, batch_size=32, lr=1e-3, samples_per_source_condition=350):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================", flush=True)
    print(f"Training Multi-Source Balanced SpectacleVisionNet CNN", flush=True)
    print(f"Device: {device} | Epochs: {epochs} | Batch: {batch_size}", flush=True)
    print(f"=======================================================", flush=True)

    # 1. Load data
    df = load_multi_source_balanced_data(samples_per_source_condition=samples_per_source_condition)
    
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["condition"]
    )
    print(f"Training set: {len(train_df):,} images | Validation set: {len(val_df):,} images", flush=True)

    # 2. Data loaders
    train_dataset = EyeImageDataset(train_df, transform=get_train_transforms())
    val_dataset = EyeImageDataset(val_df, transform=get_inference_transforms())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # 3. Model & Loss & Optimizer
    model = SpectacleVisionNet(num_spectacle_classes=2, num_conditions=4).to(device)
    
    criterion_spectacle = nn.CrossEntropyLoss()
    criterion_condition = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    history = []
    start_train_time = time.time()

    # 4. Training loop
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct_spec = 0
        total = 0

        for images, spec_labels, cond_labels in train_loader:
            images = images.to(device)
            spec_labels = spec_labels.to(device)
            cond_labels = cond_labels.to(device)

            optimizer.zero_grad()
            spec_logits, cond_logits = model(images)

            loss_spec = criterion_spectacle(spec_logits, spec_labels)
            loss_cond = criterion_condition(cond_logits, cond_labels)
            loss = 1.0 * loss_spec + 1.0 * loss_cond

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(spec_logits, dim=1)
            correct_spec += (preds == spec_labels).sum().item()
            total += images.size(0)

        scheduler.step()
        train_loss = running_loss / max(1, total)
        train_acc = (correct_spec / max(1, total)) * 100.0

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct_spec = 0
        val_correct_cond = 0
        val_normal_correct = 0
        val_normal_total = 0
        val_error_correct = 0
        val_error_total = 0
        val_total = 0

        with torch.no_grad():
            for images, spec_labels, cond_labels in val_loader:
                images = images.to(device)
                spec_labels = spec_labels.to(device)
                cond_labels = cond_labels.to(device)

                spec_logits, cond_logits = model(images)
                loss_spec = criterion_spectacle(spec_logits, spec_labels)
                loss_cond = criterion_condition(cond_logits, cond_labels)
                loss = 1.0 * loss_spec + 1.0 * loss_cond

                val_loss += loss.item() * images.size(0)
                spec_preds = torch.argmax(spec_logits, dim=1)
                cond_preds = torch.argmax(cond_logits, dim=1)

                val_correct_spec += (spec_preds == spec_labels).sum().item()
                val_correct_cond += (cond_preds == cond_labels).sum().item()
                
                for p, y in zip(spec_preds, spec_labels):
                    if y == 0:
                        val_normal_total += 1
                        if p == 0: val_normal_correct += 1
                    else:
                        val_error_total += 1
                        if p == 1: val_error_correct += 1

                val_total += images.size(0)

        val_loss = val_loss / max(1, val_total)
        val_spec_acc = (val_correct_spec / max(1, val_total)) * 100.0
        val_cond_acc = (val_correct_cond / max(1, val_total)) * 100.0
        norm_acc = (val_normal_correct / max(1, val_normal_total)) * 100.0 if val_normal_total > 0 else 0
        err_acc = (val_error_correct / max(1, val_error_total)) * 100.0 if val_error_total > 0 else 0
        epoch_sec = round(time.time() - epoch_start, 1)

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_sec}s) | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.1f}% | "
              f"Val Overall Acc: {val_spec_acc:.1f}% (Normal Acc: {norm_acc:.1f}%, Error Acc: {err_acc:.1f}%) | "
              f"Condition Acc: {val_cond_acc:.1f}%", flush=True)

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_spectacle_acc": round(val_spec_acc, 2),
            "val_normal_acc": round(norm_acc, 2),
            "val_error_acc": round(err_acc, 2),
            "val_condition_acc": round(val_cond_acc, 2)
        })

        if val_spec_acc > best_val_acc or epoch == 1:
            best_val_acc = val_spec_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ==> Saved best checkpoint to {MODEL_SAVE_PATH.name} (Val Acc: {val_spec_acc:.1f}%)", flush=True)

    total_sec = round(time.time() - start_train_time, 1)
    print(f"\nTraining Complete in {total_sec}s! Peak Validation Accuracy: {best_val_acc:.2f}%", flush=True)

    metadata = {
        "model_name": "SpectacleVisionNet_MultiSource_Balanced",
        "best_val_accuracy": round(best_val_acc, 2),
        "total_training_time_sec": total_sec,
        "classes": SPECTACLE_CLASSES,
        "conditions": CONDITION_CLASSES,
        "history": history
    }
    with open(METADATA_SAVE_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    return model, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multi-Source Balanced Spectacle Screening CNN")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--samples_per_source_condition", type=int, default=350, help="Samples per source-condition")
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        samples_per_source_condition=args.samples_per_source_condition
    )
