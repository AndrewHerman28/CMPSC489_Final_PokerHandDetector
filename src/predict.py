"""
predict.py - Phase 5
Detects cards in one photo (or a folder of photos), removes duplicate corner detections,
names the poker hand, and saves an annotated image to outputs/predictions/.

Usage:
    python3 predict.py test_photos/IMG_0001.jpg
    python3 predict.py test_photos/
"""

import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

from hand_eval import evaluate_hand

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL = PROJECT_DIR / "models" / "best.pt"
OUT_DIR = PROJECT_DIR / "outputs" / "predictions"
CONF = 0.5  # minimum confidence to keep a detection
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}  # iPhone HEIC photos: export/convert to JPG first


def detect_cards(model, image_path):
    """Returns (unique_cards, all_detections, result).
    unique_cards: {label: best_confidence}  - one entry per card, since each card's two corners
    are usually both detected with the same label."""
    result = model.predict(str(image_path), conf=CONF, device="cpu", verbose=False)[0]
    detections = []
    for box in result.boxes:
        label = model.names[int(box.cls)]
        detections.append((label, float(box.conf), [int(v) for v in box.xyxy[0].tolist()]))

    unique = {}
    for label, conf, _ in detections:
        unique[label] = max(conf, unique.get(label, 0.0))
    return unique, detections, result


def classify(unique_cards):
    """Hand name, or a message when there aren't exactly 5 different cards."""
    if len(unique_cards) != 5:
        return f"Found {len(unique_cards)} cards (need 5)"
    return evaluate_hand(list(unique_cards))


def annotate(result, detections, text, out_path):
    img = result.orig_img.copy()
    scale = max(img.shape[:2]) / 1000  # keep text readable on large phone photos
    for label, conf, (x1, y1, x2, y2) in detections:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), max(2, int(2 * scale)))
        cv2.putText(img, f"{label} {conf:.2f}", (x1, max(y1 - 6, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 * scale, (0, 255, 0), max(1, int(2 * scale)))
    # Banner with the hand name
    cv2.rectangle(img, (0, 0), (img.shape[1], int(60 * scale)), (0, 0, 0), -1)
    cv2.putText(img, text, (int(15 * scale), int(42 * scale)), cv2.FONT_HERSHEY_SIMPLEX,
                1.2 * scale, (255, 255, 255), max(2, int(3 * scale)))
    cv2.imwrite(str(out_path), img)


def run(model, image_path):
    unique, detections, result = detect_cards(model, image_path)
    text = classify(unique)
    out_path = OUT_DIR / f"{Path(image_path).stem}_pred.jpg"
    annotate(result, detections, text, out_path)
    cards = ", ".join(f"{c} ({p:.2f})" for c, p in sorted(unique.items()))
    print(f"{Path(image_path).name}: {text}  |  cards: {cards}")
    return unique, text


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return
    source = Path(sys.argv[1])
    images = sorted(p for p in source.iterdir() if p.suffix.lower() in IMG_EXTS) if source.is_dir() else [source]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL))
    for img in images:
        run(model, img)
    print(f"\nAnnotated images saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
