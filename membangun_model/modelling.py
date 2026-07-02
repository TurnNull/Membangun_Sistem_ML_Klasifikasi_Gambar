import argparse
import json
import random
from pathlib import Path

import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf


# ============================================================
# Konfigurasi reproducibility
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# Argumen command line
# ============================================================

def parse_args():
    """Membaca argumen dari command line."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default="PlantVillage_preprocessing",
        help="Path dataset hasil preprocessing."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Jumlah epoch training."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Ukuran batch."
    )

    parser.add_argument(
        "--image_size",
        type=int,
        default=224,
        help="Ukuran input gambar."
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Learning rate optimizer."
    )

    return parser.parse_args()


# ============================================================
# Fungsi dataset
# ============================================================

def load_labels(data_dir: Path) -> list:
    """Membaca daftar label dari labels.txt."""
    labels_path = data_dir / "labels.txt"

    if not labels_path.exists():
        raise FileNotFoundError(f"File labels.txt tidak ditemukan: {labels_path}")

    with open(labels_path, "r", encoding="utf-8") as file:
        labels = [line.strip() for line in file.readlines() if line.strip()]

    return labels


def load_metadata(data_dir: Path) -> dict:
    """Membaca metadata dataset jika tersedia."""
    metadata_path = data_dir / "dataset_metadata.json"

    if not metadata_path.exists():
        return {}

    with open(metadata_path, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return metadata


def build_image_datasets(data_dir: Path, image_size: int, batch_size: int):
    """Membuat dataset TensorFlow dari folder train, val, dan test."""
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"Folder train tidak ditemukan: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Folder val tidak ditemukan: {val_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Folder test tidak ditemukan: {test_dir}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="int",
        shuffle=True,
        seed=SEED
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="int",
        shuffle=False
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(image_size, image_size),
        batch_size=batch_size,
        label_mode="int",
        shuffle=False
    )

    autotune = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(autotune)
    val_ds = val_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    return train_ds, val_ds, test_ds


# ============================================================
# Fungsi model
# ============================================================

def build_model(num_classes: int, image_size: int, learning_rate: float):
    """Membangun model klasifikasi gambar berbasis MobileNetV2."""
    input_layer = tf.keras.Input(shape=(image_size, image_size, 3), name="input_gambar")

    augmentasi = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="augmentasi_data"
    )

    x = augmentasi(input_layer)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet"
    )

    # Base model dibekukan agar training lebih ringan.
    base_model.trainable = False

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout")(x)

    output_layer = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="output_klasifikasi"
    )(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ============================================================
# Pipeline training
# ============================================================

def main():
    """Menjalankan training baseline dengan MLflow autolog."""
    args = parse_args()

    data_dir = Path(args.data_dir).resolve()

    labels = load_labels(data_dir)
    metadata = load_metadata(data_dir)
    num_classes = len(labels)

    print("Dataset hasil preprocessing:", data_dir)
    print("Jumlah kelas              :", num_classes)
    print("Epoch                     :", args.epochs)
    print("Batch size                :", args.batch_size)
    print("Image size                :", args.image_size)

    train_ds, val_ds, test_ds = build_image_datasets(
        data_dir=data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size
    )

    model = build_model(
        num_classes=num_classes,
        image_size=args.image_size,
        learning_rate=args.learning_rate
    )

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("PlantVillage_Baseline_Autolog")

    # Autolog wajib untuk file modelling.py.
    mlflow.tensorflow.autolog(log_models=True)

    with mlflow.start_run(run_name="baseline_mobilenetv2_autolog"):
        mlflow.log_param("dataset_name", "PlantVillage")
        mlflow.log_param("dataset_source", "PlantVillage_preprocessing")
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("seed", SEED)

        if metadata:
            mlflow.log_dict(metadata, "dataset/dataset_metadata.json")

        labels_path = data_dir / "labels.txt"
        if labels_path.exists():
            mlflow.log_artifact(str(labels_path), artifact_path="dataset")

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs
        )

        test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

        # Evaluasi test dicatat manual agar terlihat eksplisit di MLflow.
        mlflow.log_metric("test_loss", float(test_loss))
        mlflow.log_metric("test_accuracy", float(test_accuracy))

        print("Test loss    :", test_loss)
        print("Test accuracy:", test_accuracy)


if __name__ == "__main__":
    main()