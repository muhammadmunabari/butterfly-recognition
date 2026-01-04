"""
07_prediction_visualization.py
------------------------------
Stage 7: Prediction Visualization (GoogLeNet)
"""

# ==============================
# Fix Python Path
# ==============================
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ==============================
# Import Libraries
# ==============================
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==============================
# Configuration
# ==============================
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset", "raw", "Train")
MODEL_PATH = os.path.join(PROJECT_ROOT, "results", "googlenet", "googlenet_best_model.h5")

IMG_SIZE = (224, 224)
BATCH_SIZE = 1
VALIDATION_SPLIT = 0.2
SEED = 42
NUM_SAMPLES = 9

# ==============================
# Load Model
# ==============================
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# ==============================
# Validation Generator
# ==============================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=VALIDATION_SPLIT
)

val_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=True,
    seed=SEED
)

class_labels = list(val_generator.class_indices.keys())

# ==============================
# Visualization
# ==============================
plt.figure(figsize=(12, 12))

for i in range(NUM_SAMPLES):
    image, label = next(val_generator)
    true_label = class_labels[np.argmax(label)]

    prediction = model.predict(image, verbose=0)
    pred_label = class_labels[np.argmax(prediction)]

    plt.subplot(3, 3, i + 1)
    plt.imshow(image[0])
    plt.axis("off")

    title_color = "green" if true_label == pred_label else "red"
    plt.title(
        f"True: {true_label}\nPred: {pred_label}",
        color=title_color,
        fontsize=9
    )

plt.tight_layout()
plt.show()
