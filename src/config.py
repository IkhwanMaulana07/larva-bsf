"""
Konfigurasi terpusat untuk proyek Sistem Analisis Pertumbuhan Larva BSF.
Semua path dan konstanta didefinisikan di sini agar mudah diubah dari satu tempat.
"""
from pathlib import Path

# ================= PATH DASAR =================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RUNS_DIR = PROJECT_ROOT / "runs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DATA_YAML = PROJECT_ROOT / "data.yaml"

# ================= KONFIGURASI KELAS =================
CLASS_NAMES = ["Larva Muda", "Larva Dewasa", "Prepupa", "Pupa"]
NUM_CLASSES = len(CLASS_NAMES)

# ================= KONFIGURASI UKURAN LARVA =================
# Ambang batas panjang (mm)
SIZE_THRESHOLDS = {
    "Kecil": (0, 10),
    "Sedang": (10, 18),
    "Besar": (18, 999),
}

# ================= KONFIGURASI MODEL =================
YOLO_MODEL_VARIANT = "yolo11s.pt"
IMG_SIZE = 640
DEFAULT_MODEL_PATH = MODELS_DIR / "yolov11_bsf" / "best.pt"