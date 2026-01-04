"""
Tahap 4: GoogLeNet (Inception v1) Model Definition
"""

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D,
    AveragePooling2D, GlobalAveragePooling2D,
    Dense, Dropout, concatenate
)
from tensorflow.keras.models import Model


# ==============================
# Inception Module
# ==============================
def inception_module(x, f1, f3r, f3, f5r, f5, fp):
    """
    GoogLeNet Inception Module
    """
    branch1 = Conv2D(f1, (1, 1), padding="same", activation="relu")(x)

    branch2 = Conv2D(f3r, (1, 1), padding="same", activation="relu")(x)
    branch2 = Conv2D(f3, (3, 3), padding="same", activation="relu")(branch2)

    branch3 = Conv2D(f5r, (1, 1), padding="same", activation="relu")(x)
    branch3 = Conv2D(f5, (5, 5), padding="same", activation="relu")(branch3)

    branch4 = MaxPooling2D((3, 3), strides=(1, 1), padding="same")(x)
    branch4 = Conv2D(fp, (1, 1), padding="same", activation="relu")(branch4)

    return concatenate([branch1, branch2, branch3, branch4], axis=-1)


# ==============================
# GoogLeNet Model
# ==============================
def build_googlenet(input_shape=(224, 224, 3), num_classes=50):
    inputs = Input(shape=input_shape)

    # Initial layers
    x = Conv2D(64, (7, 7), strides=(2, 2), padding="same", activation="relu")(inputs)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    x = Conv2D(64, (1, 1), activation="relu")(x)
    x = Conv2D(192, (3, 3), padding="same", activation="relu")(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # Inception blocks
    x = inception_module(x, 64, 96, 128, 16, 32, 32)
    x = inception_module(x, 128, 128, 192, 32, 96, 64)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    x = inception_module(x, 192, 96, 208, 16, 48, 64)
    x = inception_module(x, 160, 112, 224, 24, 64, 64)
    x = inception_module(x, 128, 128, 256, 24, 64, 64)
    x = inception_module(x, 112, 144, 288, 32, 64, 64)
    x = inception_module(x, 256, 160, 320, 32, 128, 128)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    x = inception_module(x, 256, 160, 320, 32, 128, 128)
    x = inception_module(x, 384, 192, 384, 48, 128, 128)

    # Classification head
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs, name="GoogLeNet")

    return model


# ==============================
# Model Summary (Debug)
# ==============================
if __name__ == "__main__":
    model = build_googlenet()
    model.summary()

