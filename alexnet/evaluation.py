"""
Tahap 6: AlexNet Model Evaluation
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# ==============================
# Configuration
# ==============================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset", "raw", "Train")
MODEL_PATH = os.path.join(PROJECT_ROOT, "results", "alexnet", "alexnet_best_model.h5")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "alexnet")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
SEED = 42

# ==============================
# Load Model
# ==============================
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully.")

# ==============================
# Validation Generator (NO augmentation)
# ==============================
val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=VALIDATION_SPLIT
)

validation_generator = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

class_labels = list(validation_generator.class_indices.keys())

# ==============================
# Prediction
# ==============================
y_true = validation_generator.classes
y_pred_probs = model.predict(validation_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

# ==============================
# Evaluation Metrics
# ==============================
report = classification_report(
    y_true,
    y_pred,
    target_names=class_labels,
    output_dict=True
)

conf_matrix = confusion_matrix(y_true, y_pred)

# ==============================
# Save Results
# ==============================
report_df = pd.DataFrame(report).transpose()
report_path = os.path.join(OUTPUT_DIR, "alexnet_classification_report.csv")
report_df.to_csv(report_path)

conf_matrix_path = os.path.join(OUTPUT_DIR, "alexnet_confusion_matrix.npy")
np.save(conf_matrix_path, conf_matrix)

print("\nEvaluation completed.")
print("Classification report saved to:", report_path)
print("Confusion matrix saved to:", conf_matrix_path)


