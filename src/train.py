"""
train.py - Phase 3
Fine-tunes YOLO26n (COCO-pretrained) on the 52-class playing card dataset using the Apple GPU (MPS).

Usage (PyCharm Terminal, from the Final folder):
    caffeinate -i python3 train.py      # caffeinate keeps the Mac awake while training

Steps:
    1) Set SMOKE_TEST = True, run once (a few minutes) to catch errors
    2) Set SMOKE_TEST = False, run the real training
    3) If it gets interrupted, set RESUME = True and run again
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"   # CPU fallback for any op MPS doesn't support

import time
import shutil
from pathlib import Path

from ultralytics import YOLO

# ---------------- Settings ----------------
SMOKE_TEST = True           # True = 1 epoch on 5% of the data, just to test the pipeline
RESUME = False              # True = continue an interrupted run from last.pt

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_YAML = PROJECT_DIR / "data" / "data.yaml"
RUNS_DIR = PROJECT_DIR / "runs"
RUN_NAME = "smoke_test" if SMOKE_TEST else "cards_v1"

EPOCHS = 1 if SMOKE_TEST else 30
FRACTION = 0.05 if SMOKE_TEST else 0.35   # share of the 21k train images (0.35 ~= 7.4k). Use 1.0 for all.
IMG_SIZE = 640
BATCH = 16                  # lower to 8 if you run out of memory
# ------------------------------------------


def main():
    run_dir = RUNS_DIR / RUN_NAME

    if RESUME:
        model = YOLO(run_dir / "weights" / "last.pt")
        start = time.time()
        model.train(resume=True)
    else:
        model = YOLO("../yolo26n.pt")            # COCO-pretrained starting weights
        start = time.time()
        model.train(
            data=str(DATA_YAML),
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH,
            fraction=FRACTION,
            device="mps",
            workers=4,
            patience=10,            # stop early if val mAP hasn't improved for 10 epochs
            fliplr=0.0,             # no horizontal flips: mirrored card text never appears in real photos
            seed=42,
            plots=True,             # saves loss curves, PR/F1 curves, confusion matrix
            project=str(RUNS_DIR),
            name=RUN_NAME,
            exist_ok=True,
        )

    minutes = (time.time() - start) / 60
    print(f"\nTraining time: {minutes:.1f} min")

    # Copy the best weights to models/ (skip for the smoke test)
    best = run_dir / "weights" / "best.pt"
    if not SMOKE_TEST and best.exists():
        models_dir = PROJECT_DIR / "models"
        models_dir.mkdir(exist_ok=True)
        shutil.copy(best, models_dir / "best.pt")
        print(f"Copied {best} -> {models_dir / 'best.pt'}")

    # Save settings + time for the slides
    with open(run_dir / "training_notes.txt", "w") as f:
        f.write(f"model=yolo26n.pt (COCO-pretrained)\nepochs={EPOCHS}\nfraction={FRACTION}\n"
                f"imgsz={IMG_SIZE}\nbatch={BATCH}\ndevice=mps\nfliplr=0.0\n"
                f"training_time_min={minutes:.1f}\n")


if __name__ == "__main__":
    main()