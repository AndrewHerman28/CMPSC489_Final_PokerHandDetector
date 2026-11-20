"""
check_dataset.py - Phase 2
1) Fixes data.yaml paths (backs up the original to data_original.yaml)
2) Counts images/labels per split and flags missing or empty label files
3) Counts boxes per class (train split) -> class balance for the slides
4) Draws labeled boxes on random training images -> outputs/label_check/
5) Copies 10 random training images -> outputs/training_samples/ (for the PPT)

Edit DATASET_DIR if your unzipped Roboflow folder is somewhere else.
"""

import random
import shutil
from pathlib import Path
from collections import Counter

import cv2
import yaml

# ---------------- Settings ----------------
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_DIR / "data"   # folder containing data.yaml
OUTPUT_DIR = PROJECT_DIR / "outputs"
N_LABEL_CHECK = 8      # images to draw boxes on
N_PPT_SAMPLES = 10     # sample training images for the PPT
SEED = 42
# ------------------------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = {"train": "train", "val": "valid", "test": "test"}


def fix_yaml():
    yaml_path = DATASET_DIR / "data.yaml"
    backup = DATASET_DIR / "data_original.yaml"
    if not backup.exists():
        shutil.copy(yaml_path, backup)

    with open(backup) as f:
        cfg = yaml.safe_load(f)

    cfg["path"] = str(DATASET_DIR)
    for key, folder in SPLITS.items():
        if (DATASET_DIR / folder / "images").exists():
            cfg[key] = f"{folder}/images"
        else:
            cfg.pop(key, None)

    # Keep only the keys Ultralytics needs (Roboflow adds extras)
    clean = {k: cfg[k] for k in ["path", "train", "val", "test", "nc", "names"] if k in cfg}
    with open(yaml_path, "w") as f:
        yaml.safe_dump(clean, f, sort_keys=False)

    names = clean["names"]
    print(f"data.yaml fixed -> path: {clean['path']}")
    print(f"Classes: {clean.get('nc', len(names))}  (first 10: {names[:10]})")
    if len(names) != 52:
        print("WARNING: expected 52 classes")
    return names


def check_split(folder):
    img_dir = DATASET_DIR / folder / "images"
    lbl_dir = DATASET_DIR / folder / "labels"
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)

    missing, empty, class_counts = [], [], Counter()
    for img in images:
        lbl = lbl_dir / (img.stem + ".txt")
        if not lbl.exists():
            missing.append(img.name)
            continue
        lines = [ln for ln in lbl.read_text().splitlines() if ln.strip()]
        if not lines:
            empty.append(img.name)
        for ln in lines:
            class_counts[int(ln.split()[0])] += 1

    print(f"{folder:>6}: {len(images):6d} images | {len(missing)} missing labels | "
          f"{len(empty)} empty labels | {sum(class_counts.values())} boxes")
    return images, class_counts


def draw_labels(images, names):
    out = OUTPUT_DIR / "label_check"
    out.mkdir(parents=True, exist_ok=True)
    lbl_dir = DATASET_DIR / "train" / "labels"

    for img_path in random.sample(images, min(N_LABEL_CHECK, len(images))):
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        lbl = lbl_dir / (img_path.stem + ".txt")
        if lbl.exists():
            for ln in lbl.read_text().splitlines():
                if not ln.strip():
                    continue
                c, cx, cy, bw, bh = ln.split()[:5]
                cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
                x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
                x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, names[int(c)], (x1, max(y1 - 5, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(str(out / img_path.name), img)
    print(f"Label check images saved to {out}  <- open these and look at what is boxed")


def copy_ppt_samples(images):
    out = OUTPUT_DIR / "training_samples"
    out.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(random.sample(images, min(N_PPT_SAMPLES, len(images))), 1):
        shutil.copy(p, out / f"sample_{i:02d}{p.suffix}")
    print(f"{N_PPT_SAMPLES} PPT sample images saved to {out}")


def main():
    random.seed(SEED)
    if not (DATASET_DIR / "data.yaml").exists():
        raise FileNotFoundError(f"No data.yaml in {DATASET_DIR} - edit DATASET_DIR at the top.")

    print("=== data.yaml ===")
    names = fix_yaml()

    print("\n=== Splits ===")
    train_imgs, train_counts = [], Counter()
    for folder in SPLITS.values():
        if (DATASET_DIR / folder / "images").exists():
            imgs, counts = check_split(folder)
            if folder == "train":
                train_imgs, train_counts = imgs, counts

    print("\n=== Train boxes per class (fewest 5 / most 5) ===")
    ranked = sorted(train_counts.items(), key=lambda kv: kv[1])
    print("Fewest:", [(names[c], n) for c, n in ranked[:5]])
    print("Most:  ", [(names[c], n) for c, n in ranked[-5:]])

    print("\n=== Visual checks ===")
    draw_labels(train_imgs, names)
    copy_ppt_samples(train_imgs)


if __name__ == "__main__":
    main()