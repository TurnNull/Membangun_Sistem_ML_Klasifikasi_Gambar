import json
import random
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image


# ============================================================
# Konfigurasi utama
# ============================================================

SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/emmarex/plantdisease"

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

RAW_DATA_DIR = PROJECT_DIR / "PlantVillage_raw"
RAW_ZIP_PATH = RAW_DATA_DIR / "PlantVillage.zip"

WORK_DIR = BASE_DIR / "PlantVillage_workdir"
NORMALIZED_DATA_DIR = WORK_DIR / "PlantVillage"

OUTPUT_DIR = BASE_DIR / "PlantVillage_preprocessing"

TRAIN_DIR = OUTPUT_DIR / "train"
VAL_DIR = OUTPUT_DIR / "val"
TEST_DIR = OUTPUT_DIR / "test"

LABELS_PATH = OUTPUT_DIR / "labels.txt"
METADATA_PATH = OUTPUT_DIR / "dataset_metadata.json"

# Jika True, folder PlantVillage_workdir akan dihapus setelah preprocessing selesai.
# Ini mencegah ukuran project membengkak karena dataset tidak tersimpan dua kali.
CLEAN_WORK_DIR_AFTER_PREPROCESSING = True


# ============================================================
# Fungsi utilitas folder dan file
# ============================================================

def reset_directory(directory: Path) -> None:
    """Menghapus folder lama lalu membuat folder baru yang kosong."""
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)


def is_image_file(file_path: Path) -> bool:
    """Memeriksa apakah file memiliki ekstensi gambar yang didukung."""
    return file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS


def download_dataset_if_needed() -> None:
    """
    Mengunduh dataset jika PlantVillage.zip belum tersedia.

    Jika dataset sudah tersedia di PlantVillage_raw, proses download akan dilewati.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_ZIP_PATH.exists():
        print(f"[INFO] File ZIP sudah tersedia: {RAW_ZIP_PATH}")
        return

    existing_images = list(RAW_DATA_DIR.rglob("*"))
    existing_images = [path for path in existing_images if is_image_file(path)]

    if existing_images:
        print("[INFO] Folder raw sudah berisi gambar. Download ZIP dilewati.")
        return

    print("[INFO] PlantVillage.zip belum ditemukan.")
    print("[INFO] Mengunduh dataset PlantVillage...")

    urllib.request.urlretrieve(DATASET_URL, RAW_ZIP_PATH)

    print(f"[INFO] Dataset berhasil diunduh ke: {RAW_ZIP_PATH}")


# ============================================================
# Fungsi ekstraksi dan deteksi root dataset
# ============================================================

def count_images_in_directory(directory: Path) -> int:
    """Menghitung jumlah file gambar dalam sebuah folder secara rekursif."""
    return sum(
        1
        for file_path in directory.rglob("*")
        if is_image_file(file_path)
    )


def has_direct_image_files(directory: Path) -> bool:
    """Mengecek apakah folder memiliki minimal satu file gambar secara langsung."""
    if not directory.is_dir():
        return False

    return any(
        file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS
        for file_path in directory.iterdir()
    )


def find_dataset_root(search_dir: Path, min_classes: int = 2) -> Path:
    """
    Mencari folder utama dataset yang berisi subfolder kelas.

    Folder dianggap sebagai root dataset jika:
    1. Memiliki minimal min_classes subfolder.
    2. Subfolder tersebut berisi file gambar.
    """
    candidates = []

    for directory in [search_dir] + [path for path in search_dir.rglob("*") if path.is_dir()]:
        subdirs = [item for item in directory.iterdir() if item.is_dir()]

        if len(subdirs) < min_classes:
            continue

        valid_class_dirs = [
            subdir
            for subdir in subdirs
            if has_direct_image_files(subdir)
        ]

        if len(valid_class_dirs) >= min_classes:
            total_images = sum(count_images_in_directory(subdir) for subdir in valid_class_dirs)

            candidates.append(
                {
                    "path": directory,
                    "total_classes": len(valid_class_dirs),
                    "total_images": total_images,
                }
            )

    if not candidates:
        raise FileNotFoundError(
            "Folder kelas tidak ditemukan. "
            "Pastikan dataset memiliki struktur folder per kelas."
        )

    # Kandidat terbaik adalah yang memiliki jumlah kelas terbanyak.
    # Jika jumlah kelas sama, pilih yang memiliki gambar paling banyak.
    candidates = sorted(
        candidates,
        key=lambda item: (item["total_classes"], item["total_images"]),
        reverse=True,
    )

    selected_dir = candidates[0]["path"]

    print(f"[INFO] Root dataset terdeteksi : {selected_dir}")
    print(f"[INFO] Jumlah kelas terdeteksi: {candidates[0]['total_classes']}")
    print(f"[INFO] Jumlah gambar terdeteksi: {candidates[0]['total_images']}")

    return selected_dir


def prepare_dataset_source() -> Path:
    """
    Menyiapkan sumber dataset tanpa membuat duplikasi folder.

    Jika PlantVillage.zip tersedia:
    - ZIP diekstrak ke folder sementara.
    - Folder kelas asli ditemukan.
    - Folder tersebut dipindahkan menjadi PlantVillage_workdir/PlantVillage.
    - Folder sementara dihapus.

    Jika ZIP tidak tersedia:
    - Dataset dibaca langsung dari PlantVillage_raw.
    """
    download_dataset_if_needed()

    if RAW_ZIP_PATH.exists():
        print("[INFO] Mode dataset: ZIP")

        reset_directory(WORK_DIR)

        temporary_extract_dir = WORK_DIR / "_temporary_extract"
        temporary_extract_dir.mkdir(parents=True, exist_ok=True)

        print("[INFO] Mengekstrak dataset ke folder sementara...")

        with zipfile.ZipFile(RAW_ZIP_PATH, "r") as zip_ref:
            zip_ref.extractall(temporary_extract_dir)

        print(f"[INFO] Dataset berhasil diekstrak sementara ke: {temporary_extract_dir}")

        dataset_root = find_dataset_root(temporary_extract_dir)

        if NORMALIZED_DATA_DIR.exists():
            shutil.rmtree(NORMALIZED_DATA_DIR)

        # Pindahkan folder dataset yang valid ke PlantVillage_workdir/PlantVillage.
        # Menggunakan move, bukan copy, agar tidak menggandakan dataset.
        shutil.move(str(dataset_root), str(NORMALIZED_DATA_DIR))

        if temporary_extract_dir.exists():
            shutil.rmtree(temporary_extract_dir)

        print(f"[INFO] Dataset final hasil ekstraksi: {NORMALIZED_DATA_DIR}")

        return NORMALIZED_DATA_DIR

    print("[INFO] Mode dataset: folder raw")

    dataset_root = find_dataset_root(RAW_DATA_DIR)
    return dataset_root


# ============================================================
# Fungsi validasi dan pengumpulan gambar
# ============================================================

def validate_image(file_path: Path) -> bool:
    """
    Memvalidasi apakah file gambar dapat dibuka dengan benar.

    Gambar rusak akan dilewati agar tidak mengganggu proses training.
    """
    try:
        with Image.open(file_path) as img:
            img.verify()

        # Dibuka ulang agar validasi lebih aman setelah verify().
        with Image.open(file_path) as img:
            img.convert("RGB")

        return True
    except Exception:
        return False


def collect_valid_images(dataset_root: Path) -> Dict[str, List[Path]]:
    """
    Mengumpulkan gambar valid dari setiap folder kelas.

    Output:
    {
        "nama_kelas": [path_gambar_1, path_gambar_2, ...]
    }
    """
    class_to_images = {}
    total_corrupt = 0

    class_dirs = sorted([item for item in dataset_root.iterdir() if item.is_dir()])

    for class_dir in class_dirs:
        class_name = class_dir.name

        image_files = [
            file_path
            for file_path in class_dir.rglob("*")
            if is_image_file(file_path)
        ]

        valid_images = []

        for image_path in image_files:
            if validate_image(image_path):
                valid_images.append(image_path)
            else:
                total_corrupt += 1
                print(f"[WARNING] Gambar rusak dilewati: {image_path}")

        if valid_images:
            class_to_images[class_name] = valid_images

    if not class_to_images:
        raise ValueError("Tidak ada gambar valid yang ditemukan pada dataset.")

    print(f"[INFO] Jumlah kelas valid : {len(class_to_images)}")
    print(f"[INFO] Jumlah gambar rusak: {total_corrupt}")

    return class_to_images


# ============================================================
# Fungsi split dan pembuatan dataset preprocessing
# ============================================================

def split_images(image_paths: List[Path]) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Membagi gambar menjadi train, validation, dan test.

    Rasio:
    train      = 70%
    validation = 15%
    test       = 15%
    """
    image_paths = image_paths.copy()
    random.shuffle(image_paths)

    total_images = len(image_paths)

    train_end = int(total_images * TRAIN_RATIO)
    val_end = train_end + int(total_images * VAL_RATIO)

    train_images = image_paths[:train_end]
    val_images = image_paths[train_end:val_end]
    test_images = image_paths[val_end:]

    return train_images, val_images, test_images


def copy_images(image_paths: List[Path], target_class_dir: Path) -> None:
    """Menyalin gambar ke folder target sesuai subset dan kelas."""
    target_class_dir.mkdir(parents=True, exist_ok=True)

    for index, source_path in enumerate(image_paths):
        target_name = f"{index:05d}_{source_path.name}"
        target_path = target_class_dir / target_name
        shutil.copy2(source_path, target_path)


def create_preprocessed_dataset(class_to_images: Dict[str, List[Path]]) -> Dict:
    """
    Membuat folder dataset hasil preprocessing.

    Struktur akhir:
    PlantVillage_preprocessing/
    ├── train/
    ├── val/
    ├── test/
    ├── labels.txt
    └── dataset_metadata.json
    """
    reset_directory(OUTPUT_DIR)

    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    metadata = {
        "dataset_name": "PlantVillage",
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "total_classes": len(class_to_images),
        "classes": [],
        "total_images": {
            "train": 0,
            "val": 0,
            "test": 0,
            "all": 0
        },
        "class_distribution": {}
    }

    labels = sorted(class_to_images.keys())

    for class_name in labels:
        image_paths = class_to_images[class_name]

        train_images, val_images, test_images = split_images(image_paths)

        copy_images(train_images, TRAIN_DIR / class_name)
        copy_images(val_images, VAL_DIR / class_name)
        copy_images(test_images, TEST_DIR / class_name)

        class_info = {
            "class_name": class_name,
            "total": len(image_paths),
            "train": len(train_images),
            "val": len(val_images),
            "test": len(test_images),
        }

        metadata["classes"].append(class_name)
        metadata["class_distribution"][class_name] = class_info

        metadata["total_images"]["train"] += len(train_images)
        metadata["total_images"]["val"] += len(val_images)
        metadata["total_images"]["test"] += len(test_images)
        metadata["total_images"]["all"] += len(image_paths)

    return metadata


# ============================================================
# Fungsi penyimpanan output
# ============================================================

def save_labels(labels: List[str]) -> None:
    """Menyimpan daftar label kelas ke file labels.txt."""
    with open(LABELS_PATH, "w", encoding="utf-8") as file:
        for label in labels:
            file.write(f"{label}\n")

    print(f"[INFO] labels.txt berhasil disimpan di: {LABELS_PATH}")


def save_metadata(metadata: Dict) -> None:
    """Menyimpan metadata preprocessing ke file JSON."""
    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, ensure_ascii=False)

    print(f"[INFO] dataset_metadata.json berhasil disimpan di: {METADATA_PATH}")


def print_summary(metadata: Dict) -> None:
    """Menampilkan ringkasan hasil preprocessing."""
    print("\n========== RINGKASAN PREPROCESSING ==========")
    print(f"Dataset          : {metadata['dataset_name']}")
    print(f"Jumlah kelas     : {metadata['total_classes']}")
    print(f"Total gambar     : {metadata['total_images']['all']}")
    print(f"Train            : {metadata['total_images']['train']}")
    print(f"Validation       : {metadata['total_images']['val']}")
    print(f"Test             : {metadata['total_images']['test']}")
    print(f"Output dataset   : {OUTPUT_DIR}")
    print("=============================================\n")


# ============================================================
# Pipeline utama
# ============================================================

def preprocess_pipeline() -> None:
    """Menjalankan seluruh pipeline preprocessing secara otomatis."""
    random.seed(SEED)

    print("[INFO] Memulai preprocessing dataset PlantVillage.")
    print(f"[INFO] PROJECT_DIR : {PROJECT_DIR}")
    print(f"[INFO] RAW_DATA_DIR: {RAW_DATA_DIR}")
    print(f"[INFO] WORK_DIR    : {WORK_DIR}")
    print(f"[INFO] OUTPUT_DIR  : {OUTPUT_DIR}")

    dataset_root = prepare_dataset_source()
    class_to_images = collect_valid_images(dataset_root)

    metadata = create_preprocessed_dataset(class_to_images)

    save_labels(metadata["classes"])
    save_metadata(metadata)
    print_summary(metadata)

    if CLEAN_WORK_DIR_AFTER_PREPROCESSING and WORK_DIR.exists():
        shutil.rmtree(WORK_DIR)
        print(f"[INFO] Folder kerja sementara dihapus: {WORK_DIR}")

    print("[INFO] Preprocessing selesai. Dataset sudah siap digunakan untuk training.")


if __name__ == "__main__":
    preprocess_pipeline()