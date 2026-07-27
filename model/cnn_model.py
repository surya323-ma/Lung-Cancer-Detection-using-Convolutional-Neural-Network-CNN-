"""
Lung Cancer Detection - CNN Architecture
=========================================
3-class classifier for lung histopathology images:
  0 -> lung_benign_tissue
  1 -> lung_adenocarcinoma
  2 -> lung_squamous_cell_carcinoma

Designed for the LC25000 lung histopathology dataset (768x768 -> resized 224x224),
but works with any image folder structured as:

    data/
      train/
        lung_benign_tissue/
        lung_adenocarcinoma/
        lung_squamous_cell_carcinoma/
      val/
        ... same structure ...

Run `python model/cnn_model.py` to train once a dataset is placed under data/.
If no dataset is present, the app falls back to demo-mode inference (see predict.py).
"""

import os

IMG_SIZE = (224, 224)
CLASS_NAMES = [
    "lung_benign_tissue",
    "lung_adenocarcinoma",
    "lung_squamous_cell_carcinoma",
]
NUM_CLASSES = len(CLASS_NAMES)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "lung_cnn.h5")


def build_model(input_shape=(224, 224, 3), num_classes=NUM_CLASSES):
    """Builds the CNN architecture. Imports TensorFlow lazily so the rest of
    the app (Flask, demo mode) can run even in environments without TF installed."""
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(1.0 / 255),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(256, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation="softmax"),
    ], name="lung_cancer_cnn")

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy", "AUC"],
    )
    return model


def train(data_dir="data", epochs=15, batch_size=32):
    """Trains the model on an image directory (train/ + val/ subfolders)."""
    import tensorflow as tf
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"No training data found at '{train_dir}'. "
            "Download the LC25000 lung histopathology dataset and arrange it as "
            "data/train/<class_name>/*.jpeg and data/val/<class_name>/*.jpeg"
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, image_size=IMG_SIZE, batch_size=batch_size, label_mode="categorical"
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir, image_size=IMG_SIZE, batch_size=batch_size, label_mode="categorical"
    )

    model = build_model()
    callbacks = [
        ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy"),
        EarlyStopping(patience=4, restore_best_weights=True),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    model.save(MODEL_PATH)
    return history


if __name__ == "__main__":
    train()
