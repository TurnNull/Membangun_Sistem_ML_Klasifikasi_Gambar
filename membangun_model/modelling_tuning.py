import argparse
import json
import os
import random
import time
from pathlib import Path

import dagshub
import matplotlib.pyplot as plt
import mlflow
import mlflow.tensorflow
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


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
    """Membaca argumen command line."""
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
        help="Jumlah epoch untuk setiap trial."
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
        "--n_trials",
        type=int,
        default=3,
        help="Jumlah trial tuning Optuna."
    )

    parser.add_argument(
        "--use_dagshub",
        action="store_true",
        help="Aktifkan tracking online menggunakan DagsHub."
    )

    return parser.parse_args()


# ============================================================
# Setup MLflow dan DagsHub
# ============================================================

def setup_mlflow(use_dagshub: bool) -> None:
    """
    Mengatur MLflow Tracking.

    Jika use_dagshub aktif, script membaca:
    - DAGSHUB_REPO_OWNER
    - DAGSHUB_REPO_NAME

    Jika tidak aktif, tracking disimpan lokal di ./mlruns.
    """
    if use_dagshub:
        repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
        repo_name = os.getenv("DAGSHUB_REPO_NAME")

        if not repo_owner or not repo_name:
            raise ValueError(
                "DAGSHUB_REPO_OWNER dan DAGSHUB_REPO_NAME belum diatur."
            )

        dagshub.init(
            repo_owner=repo_owner,
            repo_name=repo_name,
            mlflow=True
        )

        print("[INFO] MLflow Tracking menggunakan DagsHub.")
        print("[INFO] DagsHub owner:", repo_owner)
        print("[INFO] DagsHub repo :", repo_name)
    else:
        mlflow.set_tracking_uri("file:./mlruns")
        print("[INFO] MLflow Tracking menggunakan lokal ./mlruns.")

    mlflow.set_experiment("PlantVillage_Tuning_Manual_Logging")


# ============================================================
# Dataset
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
    """Membaca metadata preprocessing jika tersedia."""
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
# Model
# ============================================================

def build_model(
    num_classes: int,
    image_size: int,
    learning_rate: float,
    dropout_rate: float
):
    """Membangun model MobileNetV2 untuk klasifikasi gambar."""
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

    base_model.trainable = False

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="dropout")(x)

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
# Evaluasi dan artefak
# ============================================================

def collect_predictions(model, dataset):
    """Mengumpulkan label asli, label prediksi, confidence, dan latency."""
    y_true = []
    y_pred = []
    y_confidence = []

    total_images = 0
    start_time = time.time()

    for images, labels in dataset:
        probabilities = model.predict(images, verbose=0)
        predictions = np.argmax(probabilities, axis=1)
        confidence = np.max(probabilities, axis=1)

        y_true.extend(labels.numpy().tolist())
        y_pred.extend(predictions.tolist())
        y_confidence.extend(confidence.tolist())

        total_images += images.shape[0]

    elapsed_time = time.time() - start_time
    latency_ms_per_image = (elapsed_time / max(total_images, 1)) * 1000

    return (
        np.array(y_true),
        np.array(y_pred),
        np.array(y_confidence),
        latency_ms_per_image
    )


def save_training_history(history, output_path: Path) -> None:
    """Menyimpan grafik riwayat training."""
    plt.figure(figsize=(10, 6))

    for metric_name, values in history.history.items():
        plt.plot(values, label=metric_name)

    plt.title("Riwayat Training Model")
    plt.xlabel("Epoch")
    plt.ylabel("Nilai")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_confusion_matrix(y_true, y_pred, labels, output_path: Path) -> None:
    """Menyimpan confusion matrix sebagai gambar."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(14, 12))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()

    tick_positions = np.arange(len(labels))
    plt.xticks(tick_positions, labels, rotation=90, fontsize=7)
    plt.yticks(tick_positions, labels, fontsize=7)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_classification_report(y_true, y_pred, labels, output_path: Path) -> None:
    """Menyimpan classification report ke CSV."""
    report = classification_report(
        y_true,
        y_pred,
        target_names=labels,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(output_path, index=True)


def save_model_summary(model, output_path: Path) -> None:
    """Menyimpan ringkasan arsitektur model."""
    with open(output_path, "w", encoding="utf-8") as file:
        model.summary(print_fn=lambda line: file.write(line + "\n"))


def save_labels_artifact(labels, output_path: Path) -> None:
    """Menyimpan label kelas sebagai artefak."""
    with open(output_path, "w", encoding="utf-8") as file:
        for label in labels:
            file.write(label + "\n")


def get_file_size_mb(path: Path) -> float:
    """Menghitung ukuran file dalam MB."""
    if not path.exists():
        return 0.0

    return path.stat().st_size / (1024 * 1024)


def log_history_metrics(history) -> None:
    """Manual logging seluruh metrik training per epoch."""
    for metric_name, values in history.history.items():
        for epoch, value in enumerate(values, start=1):
            mlflow.log_metric(metric_name, float(value), step=epoch)


# ============================================================
# Objective Optuna
# ============================================================

def objective(
    trial,
    args,
    labels,
    metadata,
    train_ds,
    val_ds,
    test_ds,
    artifacts_dir: Path
):
    """Fungsi objektif untuk hyperparameter tuning."""
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.5)

    num_classes = len(labels)

    run_name = f"trial_{trial.number}_manual_logging"

    with mlflow.start_run(run_name=run_name):
        # Manual logging parameter.
        mlflow.log_param("model_name", "MobileNetV2")
        mlflow.log_param("dataset_name", "PlantVillage")
        mlflow.log_param("dataset_source", "PlantVillage_preprocessing")
        mlflow.log_param("num_classes", num_classes)
        mlflow.log_param("image_size", args.image_size)
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("optimizer", "Adam")
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("dropout_rate", dropout_rate)
        mlflow.log_param("base_model_trainable", False)
        mlflow.log_param("tuning_method", "Optuna")
        mlflow.log_param("seed", SEED)

        if metadata:
            mlflow.log_dict(metadata, "dataset/dataset_metadata.json")

        model = build_model(
            num_classes=num_classes,
            image_size=args.image_size,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate
        )

        start_training = time.time()

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.epochs,
            verbose=1
        )

        training_time_seconds = time.time() - start_training

        test_loss, test_accuracy = model.evaluate(test_ds, verbose=1)

        y_true, y_pred, y_confidence, latency_ms = collect_predictions(model, test_ds)

        precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
        average_confidence = float(np.mean(y_confidence))

        # Manual logging metrik seperti autolog.
        log_history_metrics(history)

        # Manual logging metrik tambahan untuk advance.
        mlflow.log_metric("test_loss", float(test_loss))
        mlflow.log_metric("test_accuracy", float(test_accuracy))
        mlflow.log_metric("precision_macro", float(precision_macro))
        mlflow.log_metric("recall_macro", float(recall_macro))
        mlflow.log_metric("f1_macro", float(f1_macro))
        mlflow.log_metric("average_prediction_confidence", average_confidence)
        mlflow.log_metric("inference_latency_ms_per_image", float(latency_ms))
        mlflow.log_metric("training_time_seconds", float(training_time_seconds))

        trial_dir = artifacts_dir / f"trial_{trial.number}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        training_history_path = trial_dir / "training_history.png"
        confusion_matrix_path = trial_dir / "confusion_matrix.png"
        classification_report_path = trial_dir / "classification_report.csv"
        model_summary_path = trial_dir / "model_summary.txt"
        labels_path = trial_dir / "labels.txt"
        trial_params_path = trial_dir / "trial_params.json"
        keras_model_path = trial_dir / "model.keras"

        save_training_history(history, training_history_path)
        save_confusion_matrix(y_true, y_pred, labels, confusion_matrix_path)
        save_classification_report(y_true, y_pred, labels, classification_report_path)
        save_model_summary(model, model_summary_path)
        save_labels_artifact(labels, labels_path)

        with open(trial_params_path, "w", encoding="utf-8") as file:
            json.dump(trial.params, file, indent=4)

        model.save(keras_model_path)
        model_size_mb = get_file_size_mb(keras_model_path)
        mlflow.log_metric("model_size_mb", float(model_size_mb))

        # Artefak tambahan untuk kriteria advance.
        mlflow.log_artifact(str(training_history_path), artifact_path="plots")
        mlflow.log_artifact(str(confusion_matrix_path), artifact_path="plots")
        mlflow.log_artifact(str(classification_report_path), artifact_path="reports")
        mlflow.log_artifact(str(model_summary_path), artifact_path="model_info")
        mlflow.log_artifact(str(labels_path), artifact_path="dataset")
        mlflow.log_artifact(str(trial_params_path), artifact_path="params")
        mlflow.log_artifact(str(keras_model_path), artifact_path="keras_model_file")

        # Model utama dicatat ke MLflow.
        mlflow.tensorflow.log_model(model, artifact_path="model")

        best_val_accuracy = max(history.history["val_accuracy"])

        print("\n========== HASIL TRIAL ==========")
        print("Trial              :", trial.number)
        print("Val accuracy terbaik:", best_val_accuracy)
        print("Test accuracy      :", test_accuracy)
        print("F1 macro           :", f1_macro)
        print("=================================\n")

        return best_val_accuracy


# ============================================================
# Pipeline utama
# ============================================================

def main():
    """Menjalankan tuning dan manual logging MLflow."""
    args = parse_args()

    data_dir = Path(args.data_dir).resolve()
    artifacts_dir = Path("artifacts_tuning").resolve()
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    setup_mlflow(use_dagshub=args.use_dagshub)

    labels = load_labels(data_dir)
    metadata = load_metadata(data_dir)

    train_ds, val_ds, test_ds = build_image_datasets(
        data_dir=data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size
    )

    study = optuna.create_study(direction="maximize")

    study.optimize(
        lambda trial: objective(
            trial=trial,
            args=args,
            labels=labels,
            metadata=metadata,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
            artifacts_dir=artifacts_dir
        ),
        n_trials=args.n_trials
    )

    best_params_path = artifacts_dir / "best_params.json"

    with open(best_params_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "best_trial": study.best_trial.number,
                "best_value": study.best_value,
                "best_params": study.best_params,
            },
            file,
            indent=4
        )

    print("\n========== HASIL TUNING TERBAIK ==========")
    print("Best trial :", study.best_trial.number)
    print("Best value :", study.best_value)
    print("Best params:", study.best_params)
    print("File best params:", best_params_path)
    print("==========================================\n")


if __name__ == "__main__":
    main()