"""
env_check.py - Phase 1
Imports every library the project uses and prints versions for the Materials slide.

Install first (PyCharm Terminal, with the project venv active):
    1) PyTorch (macOS): pip3 install torch torchvision
       (Apple Silicon Macs get GPU support via MPS automatically)
    2) pip install ultralytics opencv-python matplotlib pandas pyyaml
"""

# --- Standard library ---
import os
import sys
import csv
import random
import shutil
import platform
from pathlib import Path
from collections import Counter

# --- Third-party ---
import numpy as np
import pandas as pd
import cv2
import yaml
import matplotlib
import matplotlib.pyplot as plt
import torch
import ultralytics
from ultralytics import YOLO


def main():
    print("=== Environment check ===")
    print(f"OS:            {platform.system()} {platform.release()}")
    print(f"Python:        {sys.version.split()[0]}")
    print(f"PyTorch:       {torch.__version__}")
    print(f"Ultralytics:   {ultralytics.__version__}")
    print(f"OpenCV:        {cv2.__version__}")
    print(f"NumPy:         {np.__version__}")
    print(f"Pandas:        {pd.__version__}")
    print(f"Matplotlib:    {matplotlib.__version__}")
    print(f"PyYAML:        {yaml.__version__}")

    print("\n=== GPU check ===")
    if torch.cuda.is_available():
        print(f"CUDA available: True (CUDA {torch.version.cuda})")
        print(f"GPU:            {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM:           {vram_gb:.1f} GB")
    elif torch.backends.mps.is_available():
        print("Apple Silicon GPU (MPS) available: True")
        print(f"Chip: {platform.processor() or platform.machine()}")
        print('Use device="mps" when training.')
    else:
        print("No GPU found (no CUDA, no MPS) -> training would run on CPU (too slow for 24k images).")
        print("Fix: do the training step in Google Colab, then copy best.pt back.")

    # Quick smoke test: load the pretrained model (downloads yolo26n.pt the first time)
    print("\n=== Model load test ===")
    model = YOLO("../yolo26n.pt")
    print(f"Loaded {model.model_name if hasattr(model, 'model_name') else 'yolo26n.pt'} OK")


if __name__ == "__main__":
    main()