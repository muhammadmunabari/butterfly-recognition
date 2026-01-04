"""
04_model_definition.py
----------------------
Stage 4: AlexNet Architecture Definition

This script defines the AlexNet CNN architecture
adapted for butterfly species classification.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Flatten,
    Dropout
)

# ==============================
# Configuration
# ==============================
INPUT_SHAPE = (224, 224, 3)
NUM_CLASSES = 50

# ==============================
# AlexNet Model Definition
# ==============================
def build_alexnet(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    model = Sequential(name="AlexNet")

    # Layer 1
    model.add(Conv2D(96, (11, 11), strides=4, activation="relu",
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

    # Layer 2
    model.add(Conv2D(256, (5, 5), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

    # Layer 3
    model.add(Conv2D(384, (3, 3), padding="same", activation="relu"))

    # Layer 4
    model.add(Conv2D(384, (3, 3), padding="same", activation="relu"))

    # Layer 5
    model.add(Conv2D(256, (3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation="relu"))
    model.add(Dropout(0.5))

    # Output Layer
    model.add(Dense(num_classes, activation="softmax"))

    return model


# ==============================
# Model Summary
# ==============================
if __name__ == "__main__":
    alexnet_model = build_alexnet()
    alexnet_model.summary()
