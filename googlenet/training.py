"""
Tahap 5: GoogLeNet Model Training
"""

# ==============================
# PYTHON PATH
# ==============================
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ==============================
# Import Libraries
# ==============================
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

from common.preprocessing import train_generator, validation_generator
from googlenet.model_definition import build_googlenet

# ==============================
# Configuration
# ==============================
RESULT_DIR = os.path.join(PROJECT_ROOT, "results", "googlenet")
os.makedirs(RESULT_DIR, exist_ok=True)

MODEL_PATH = os.path.join(RESULT_DIR, "googlenet_best_model.h5")
LOG_PATH = os.path.join(RESULT_DIR, "googlenet_training_log.csv")

EPOCHS = 30
LEARNING_RATE = 1e-4

# ==============================
# Build & Compile Model
# ==============================
model = build_googlenet()

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==============================
# Callbacks
# ==============================
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    CSVLogger(LOG_PATH)
]

# ==============================
# Training
# ==============================
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("\nTraining selesai.")
print("Model terbaik disimpan di:", MODEL_PATH)
print("Training log disimpan di:", LOG_PATH)

