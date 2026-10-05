"""
==============================================================================
AI Vision Screening: Dataset CSV Generator
==============================================================================
Scans the 3 datasets:
1. AI generated Dataset (astigmatism, hyperopia, myopia, normal)
2. Real time dataset (Myopia_images, normal)
3. eye_dataset (hyperopia, myopia, normal)

Generates CSVs into: backend/dataset/csv/
- ai_generated_dataset.csv
- real_time_dataset.csv
- eye_dataset.csv
- master_dataset.csv
- dataset_summary.csv
=============================================================================="""

import os
import pandas as pd
from pathlib import Path

# Paths
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent.parent
WORKSPACE_DATASETS_DIR = PROJECT_ROOT / "dataset"
BACKEND_DATASETS_DIR = BACKEND_DIR / "dataset"

# Output CSV folder: backend/dataset/csv/
CSV_OUTPUT_DIR = BACKEND_DIR / "dataset" / "csv"
CSV_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

def get_normalized_condition(folder_name: str) -> tuple[str, int, str]:
    """
    Returns: (condition_name, spectacles_needed (0 or 1), display_label)
    """
    lower = folder_name.lower().strip()
    if "normal" in lower or "emmetrop" in lower:
        return ("normal", 0, "No Spectacles Needed")
    elif "myopia" in lower:
        return ("myopia", 1, "Spectacles Required")
    elif "hyperopia" in lower:
        return ("hyperopia", 1, "Spectacles Required")
    elif "astigmatism" in lower:
        return ("astigmatism", 1, "Spectacles Required")
    else:
        return (folder_name, 1, "Spectacles Required")

def scan_directory(dataset_dir: Path, dataset_name: str) -> list[dict]:
    records = []
    if not dataset_dir.exists():
        print(f"[Warning] Directory does not exist: {dataset_dir}")
        return records

    # Iterate subfolders (categories)
    for category_dir in sorted(dataset_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        category_name = category_dir.name
        condition, spectacles_needed, label = get_normalized_condition(category_name)

        # Iterate image files
        for img_file in sorted(category_dir.iterdir()):
            if img_file.is_file() and img_file.suffix.lower() in VALID_EXTENSIONS:
                # Relative path from project root for portability
                try:
                    rel_path = img_file.relative_to(PROJECT_ROOT).as_posix()
                except ValueError:
                    rel_path = img_file.as_posix()
                    
                records.append({
                    "filename": img_file.name,
                    "dataset_source": dataset_name,
                    "folder_category": category_name,
                    "condition": condition,
                    "is_refractive_error": 1 if spectacles_needed == 1 else 0,
                    "spectacles_needed": spectacles_needed,
                    "label": label,
                    "relative_path": rel_path,
                    "absolute_path": str(img_file.resolve())
                })
    return records

def generate_csvs():
    print("=" * 60)
    print("Scanning and Generating Dataset CSV Files...")
    print("=" * 60)

    # Prefer workspace dataset root, fallback to backend dataset directory
    datasets_to_scan = [
        ("AI generated Dataset", [WORKSPACE_DATASETS_DIR / "AI generated Dataset", BACKEND_DATASETS_DIR / "AI generated Dataset"]),
        ("Real time dataset", [WORKSPACE_DATASETS_DIR / "Real time dataset", BACKEND_DATASETS_DIR / "Real time dataset"]),
        ("eye_dataset", [WORKSPACE_DATASETS_DIR / "eye_dataset", PROJECT_ROOT / "eye_dataset", BACKEND_DATASETS_DIR / "eye_dataset"])
    ]

    all_records = []
    csv_outputs = {}

    for name, candidate_dirs in datasets_to_scan:
        chosen_dir = None
        for c_dir in candidate_dirs:
            if c_dir.exists() and any(c_dir.iterdir()):
                chosen_dir = c_dir
                break
        
        if not chosen_dir:
            print(f"[Error] Could not find valid folder for {name}")
            continue

        print(f"Scanning '{name}' at: {chosen_dir}")
        records = scan_directory(chosen_dir, name)
        df_ds = pd.DataFrame(records)
        all_records.extend(records)
        
        # Save individual CSV to dataset/csv/
        safe_name = name.lower().replace(" ", "_") + ".csv"
        csv_path = CSV_OUTPUT_DIR / safe_name
        df_ds.to_csv(csv_path, index=False)
        csv_outputs[name] = csv_path
        print(f" -> Saved {len(df_ds):,} records to {csv_path.name}")
        print(df_ds.groupby(["condition", "label"]).size())
        print("-" * 40)

    # Master combined CSV
    df_master = pd.DataFrame(all_records)
    master_csv_path = CSV_OUTPUT_DIR / "master_dataset.csv"
    df_master.to_csv(master_csv_path, index=False)
    print(f"\n[Success] Generated Master CSV: {master_csv_path} with {len(df_master):,} total images.")

    # Summary table CSV
    summary_df = df_master.groupby(["dataset_source", "condition", "label"]).size().reset_index(name="image_count")
    summary_csv_path = CSV_OUTPUT_DIR / "dataset_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"[Success] Generated Dataset Summary CSV: {summary_csv_path.name}\n")
    print(summary_df.to_string(index=False))
    print("=" * 60)

if __name__ == "__main__":
    generate_csvs()
