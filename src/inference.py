"""
Modul inti untuk analisis larva BSF: deteksi, ekstraksi ciri visual, dan rekomendasi panen.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import cv2
import numpy as np
from pathlib import Path
from collections import Counter
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolov11_bsf" / "best.pt"
CLASS_NAMES = ["Larva Muda", "Larva Dewasa", "Prepupa", "Pupa"]

_model = None

def get_model():
    global _model
    if _model is None:
        _model = YOLO(str(MODEL_PATH))
    return _model


def detect_larvae(image_path, conf_threshold=0.25):
    model = get_model()
    with torch.no_grad():
        results = model.predict(source=str(image_path), conf=conf_threshold, verbose=False)
    result = results[0]
    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "class_id": cls_id,
            "class_name": CLASS_NAMES[cls_id],
            "confidence": conf,
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
        })
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return detections, result.orig_img


def visualize_detection_result(image_path, conf_threshold=0.25):
    """
    Menjalankan deteksi dan mengembalikan citra RGB dengan bounding box
    serta jumlah deteksi.
    """
    model = get_model()
    results = model.predict(source=str(image_path), conf=conf_threshold, verbose=False)
    result = results[0]
    
    img_bgr = result.plot()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    n_det = len(result.boxes)
    
    return img_rgb, n_det



def get_dominant_color(crop_img, k=3):
    img_small = cv2.resize(crop_img, (50, 50))
    pixels = img_small.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    counts = np.bincount(labels.flatten())
    dominant_bgr = centers[np.argmax(counts)]
    b, g, r = dominant_bgr
    return int(r), int(g), int(b)


def classify_color_name(r, g, b):
    brightness = (r + g + b) / 3
    if brightness < 60:
        return "hitam pekat"
    elif r > 150 and g > 100 and b < 100:
        return "oranye kecoklatan"
    elif r > 180 and g > 150 and b > 100:
        return "krem pucat"
    elif r > 100 and g > 70 and b < 90:
        return "coklat kemerahan"
    elif brightness > 150:
        return "pucat/terang"
    else:
        return "coklat gelap"


def analyze_texture(crop_img):
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    if laplacian_var > 500:
        texture_label = "kasar/berkerak"
    elif laplacian_var > 150:
        texture_label = "bergaris jelas (segmen tubuh terlihat)"
    else:
        texture_label = "halus"
    return {"laplacian_var": laplacian_var, "edge_density": edge_density, "texture_label": texture_label}


def count_larvae_per_phase(detections):
    phase_counts = Counter(det["class_name"] for det in detections)
    result = {name: phase_counts.get(name, 0) for name in CLASS_NAMES}
    result["Total"] = sum(result.values())
    return result


def generate_harvest_recommendation(phase_counts):
    total = phase_counts.get("Total", sum(phase_counts.get(name, 0) for name in CLASS_NAMES))
    if total == 0:
        return {"status": "Tidak Ada Data", "rekomendasi": "Tidak ada larva terdeteksi pada gambar.", "urgensi": "none"}

    pct = {name: phase_counts.get(name, 0) / total * 100 for name in CLASS_NAMES}
    dominant_phase = max(pct, key=pct.get)
    dominant_pct = pct[dominant_phase]

    if dominant_phase == "Larva Muda" and dominant_pct > 40:
        return {"status": "Belum Waktunya Panen",
                "rekomendasi": f"Mayoritas larva masih dalam fase Larva Muda ({dominant_pct:.1f}%). Disarankan menunggu 7-10 hari lagi.",
                "urgensi": "low", "estimasi_hari": "7-10 hari"}
    elif dominant_phase == "Larva Dewasa" and dominant_pct > 35:
        return {"status": "Mendekati Waktu Panen",
                "rekomendasi": f"Mayoritas larva dalam fase Larva Dewasa ({dominant_pct:.1f}%). Pantau tiap 2-3 hari.",
                "urgensi": "medium", "estimasi_hari": "3-5 hari"}
    elif dominant_phase == "Prepupa" and dominant_pct > 25:
        return {"status": "WAKTU PANEN OPTIMAL",
                "rekomendasi": f"Larva didominasi fase Prepupa ({dominant_pct:.1f}%). Disarankan panen SEGERA.",
                "urgensi": "high", "estimasi_hari": "1-2 hari"}
    elif dominant_phase == "Pupa" and dominant_pct > 20:
        return {"status": "Waktu Panen Terlewat",
                "rekomendasi": f"Sebagian larva sudah menjadi Pupa ({dominant_pct:.1f}%). Segera panen sisa Prepupa.",
                "urgensi": "critical", "estimasi_hari": "Segera"}
    else:
        return {"status": "Kondisi Campuran",
                "rekomendasi": f"Populasi tersebar merata (dominan: {dominant_phase} {dominant_pct:.1f}%). Monitor berkala.",
                "urgensi": "medium", "estimasi_hari": "Monitor 3-5 hari"}


def full_analysis_pipeline(image_path, conf_threshold=0.25):
    detections, orig_img = detect_larvae(image_path, conf_threshold)
    larvae_details = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        crop = orig_img[max(y1, 0):y2, max(x1, 0):x2]
        traits = {}
        if crop.size > 0:
            r, g, b = get_dominant_color(crop)
            traits = {"color_name": classify_color_name(r, g, b),
                      "texture_label": analyze_texture(crop)["texture_label"]}
        larvae_details.append({
            "class_name": det["class_name"],
            "confidence": round(det["confidence"], 4),
            "bbox": det["bbox"],
            **traits,
        })
    phase_counts = count_larvae_per_phase(detections)
    recommendation = generate_harvest_recommendation(phase_counts)
    return {
        "total_larvae_detected": phase_counts["Total"],
        "phase_counts": {name: phase_counts[name] for name in CLASS_NAMES},
        "larvae_details": larvae_details,
        "harvest_recommendation": recommendation,
    }
