"""
05_training.py
--------------
Stage 5: AlexNet Model Compilation & Training
Self-contained training script (journal-safe).
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, Dropout
)

# ==============================
# Paths & Configuration
# ==============================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset", "raw", "Train")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results", "alexnet")
os.makedirs(RESULT_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-4
VALIDATION_SPLIT = 0.2
SEED = 42
NUM_CLASSES = 50

MODEL_PATH = os.path.join(RESULT_DIR, "alexnet_best_model.h5")
LOG_PATH = os.path.join(RESULT_DIR, "alexnet_training_log.csv")

# ==============================
# Data Generators
# ==============================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=VALIDATION_SPLIT
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=VALIDATION_SPLIT
)

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
# AlexNet Definition
# ==============================
model = Sequential(name="AlexNet")

model.add(Conv2D(96, (11, 11), strides=4, activation="relu",
                 input_shape=(224, 224, 3)))
model.add(MaxPooling2D(3, strides=2))

model.add(Conv2D(256, (5, 5), padding="same", activation="relu"))
model.add(MaxPooling2D(3, strides=2))

model.add(Conv2D(384, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(384, (3, 3), padding="same", activation="relu"))
model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(3, strides=2))

model.add(Flatten())
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(4096, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation="softmax"))

# ==============================
# Compile Model
# ==============================
model.compile(
    optimizer=Adam(learning_rate=LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==============================
# Callbacks
# ==============================
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True),
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
