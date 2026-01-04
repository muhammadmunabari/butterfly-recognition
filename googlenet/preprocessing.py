"""
03_preprocessing.py
-------------------
Stage 3: Image Preprocessing & Data Generator

This script prepares the butterfly dataset for CNN training by:
- Resizing images
- Normalizing pixel values
- Applying data augmentation
- Creating training and validation generators
"""

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==============================
# Configuration
# ==============================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset", "raw", "Train")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
SEED = 42

# ==============================
# Data Generators
# ==============================

# Training data generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=VALIDATION_SPLIT
)

# Validation data generator (no augmentation)
val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=VALIDATION_SPLIT
)

# ==============================
# Generator Construction
# ==============================
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=SEED
)

validation_generator = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=SEED
)

# ==============================
# Dataset Information
# ==============================
NUM_CLASSES = train_generator.num_classes
TRAIN_SAMPLES = train_generator.samples
VAL_SAMPLES = validation_generator.samples

print("\nPreprocessing Summary")
print("----------------------------")
print(f"Image Size        : {IMG_SIZE}")
print(f"Batch Size        : {BATCH_SIZE}")
print(f"Training Samples  : {TRAIN_SAMPLES}")
print(f"Validation Samples: {VAL_SAMPLES}")
print(f"Number of Classes : {NUM_CLASSES}")
