"""
02_dataset_validation.py
------------------------
Stage 2: Dataset Validation & Exploration

This script validates the butterfly dataset by analyzing:
- Number of classes
- Number of images per class
- Dataset balance statistics

Outputs are saved for journal documentation.
"""

import os
import pandas as pd

# ==============================
# Configuration
# ==============================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset", "raw", "Train")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "dataset_analysis")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================
# Dataset Validation
# ==============================
class_names = [
    folder for folder in os.listdir(DATASET_DIR)
    if os.path.isdir(os.path.join(DATASET_DIR, folder))
]

data_summary = []

for class_name in class_names:
    class_path = os.path.join(DATASET_DIR, class_name)
    image_files = [
        img for img in os.listdir(class_path)
        if img.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    data_summary.append({
        "class_name": class_name,
        "num_images": len(image_files)
    })

df = pd.DataFrame(data_summary)
df = df.sort_values(by="num_images", ascending=False)

# ==============================
# Statistics
# ==============================
total_classes = df.shape[0]
total_images = df["num_images"].sum()
min_images = df["num_images"].min()
max_images = df["num_images"].max()
mean_images = df["num_images"].mean()

print("Dataset Validation Summary")
print("----------------------------")
print(f"Total Classes       : {total_classes}")
print(f"Total Images        : {total_images}")
print(f"Images per Class    : min={min_images}, max={max_images}, mean={mean_images:.2f}")

# ==============================
# Save Results
# ==============================
csv_path = os.path.join(OUTPUT_DIR, "dataset_class_distribution.csv")
df.to_csv(csv_path, index=False)

print("\nClass distribution saved to:")
print(csv_path)

# ==============================
# Imbalance Warning
# ==============================
imbalance_threshold = mean_images * 0.5
imbalanced_classes = df[df["num_images"] < imbalance_threshold]

if not imbalanced_classes.empty:
    print("\nWarning: Potential class imbalance detected.")
    print("Classes with low sample size:")
    print(imbalanced_classes.head())
else:
    print("\nNo significant class imbalance detected.")
