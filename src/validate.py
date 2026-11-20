"""
validate.py - Phase 4
Evaluates models/best.pt on the held-out TEST split (1,010 images never used in training).
Saves PR/F1/P/R curves + confusion matrix to runs/test_eval/ and prints the most-confused card pairs.

Runs on CPU (MPS hits the same macOS 13.2 bug as training). Takes a few minutes.
"""

from pathlib import Path

import numpy as np
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL = PROJECT_DIR / "models" / "best.pt"
DATA_YAML = PROJECT_DIR / "data" / "data.yaml"
TOP_CONFUSIONS = 10


def main():
    model = YOLO(str(MODEL))
    m = model.val(data=str(DATA_YAML), split="test", imgsz=640, batch=16, device="cpu",
                  plots=True, project=str(PROJECT_DIR / "runs"), name="test_eval", exist_ok=True)

    print("\n=== Test split results (for the Results slide) ===")
    print(f"Precision:  {m.box.mp:.3f}")
    print(f"Recall:     {m.box.mr:.3f}")
    print(f"mAP50:      {m.box.map50:.3f}")
    print(f"mAP50-95:   {m.box.map:.3f}")

    # Per-class mAP50-95, weakest 5
    names = model.names
    per_class = sorted(((names[c], ap) for c, ap in zip(m.box.ap_class_index, m.box.maps[m.box.ap_class_index])),
                       key=lambda x: x[1])
    print("\nWeakest 5 classes (mAP50-95):", [(n, round(float(a), 3)) for n, a in per_class[:5]])

    # Most-confused pairs. Ultralytics matrix: rows = predicted, cols = true; last index = background
    cm = m.confusion_matrix.matrix
    nc = len(names)
    pairs = []
    for pred in range(nc):
        for true in range(nc):
            if pred != true and cm[pred, true] > 0:
                pairs.append((int(cm[pred, true]), names[true], names[pred]))
    pairs.sort(reverse=True)
    print(f"\nTop {TOP_CONFUSIONS} confusions (count, TRUE card -> PREDICTED as):")
    for count, true, pred in pairs[:TOP_CONFUSIONS]:
        print(f"  {count:4d}  {true:>3} -> {pred}")
    if cm.shape[0] > nc:
        missed = int(cm[nc, :nc].sum())       # true card, nothing detected
        false_pos = int(cm[:nc, nc].sum())    # detection where there is no card
        print(f"\nMissed cards (false negatives): {missed}")
        print(f"Detections with no real card (false positives): {false_pos}")
    print(f"\nPlots saved to {PROJECT_DIR / 'runs' / 'test_eval'}")


if __name__ == "__main__":
    main()