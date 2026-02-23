"""
=============================================================
Spinach Defect Detection — Dataset Preparation Script
=============================================================
Run from your Spinach_Project folder:
  python 01_prepare_dataset.py
=============================================================
"""

import os
import re
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

RANDOM_SEED = 42
BASE_DIR    = Path.cwd()
RAW_DIR     = BASE_DIR / "data" / "raw"
SPLITS_DIR  = BASE_DIR / "data" / "splits"

CLASS_NAMES = [
    "spinach_leaf",   # 0
    "stem",           # 1
    "GOOD",           # 2
    "YELLOW",         # 3
    "HOLE",           # 4
    "TRACK",          # 5
    "WSPOT",          # 6
    "FSPOT",          # 7
]

def get_class_id(folder_name):
    """
    Maps a folder name to our class ID.
    Strips trailing (NNN) count from Mendeley folder names e.g. 'Anthracnose(102)' -> 'Anthracnose'
    Then does case-insensitive keyword matching.
    """
    # Strip trailing (number) e.g. "Anthracnose(102)" -> "Anthracnose"
    clean = re.sub(r'\s*\(\d+\)\s*$', '', folder_name).strip().lower()

    # Dataset 1 — Malabar spinach
    if clean in ("healthy",):
        return 2   # GOOD
    if "pest" in clean or "damage" in clean or "hole" in clean:
        return 4   # HOLE
    if "bacterial" in clean:
        return 7   # FSPOT
    if "anthracnose" in clean:
        return 7   # FSPOT
    if "downy" in clean or "mildew" in clean:
        return 6   # WSPOT
    if "white" in clean and "spot" in clean:
        return 6   # WSPOT

    # Dataset 2 — Begomovirus / yellow leaf
    if "yellow" in clean or "infected" in clean or "begomovirus" in clean:
        return 3   # YELLOW
    if "healthy" in clean:
        return 2   # GOOD

    return None  # unmapped — will be skipped


# ─────────────────────────────────────────────
def create_folders():
    print("=" * 55)
    print(" STEP 1: Creating folder structure")
    print("=" * 55)
    for split in ["train", "val", "test"]:
        (SPLITS_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (SPLITS_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "runs").mkdir(exist_ok=True)
    (BASE_DIR / "reports").mkdir(exist_ok=True)
    print(f"  ✅ Base    : {BASE_DIR}")
    print(f"  ✅ Raw     : {RAW_DIR}")
    print(f"  ✅ Splits  : {SPLITS_DIR}")


# ─────────────────────────────────────────────
def scan_raw_data():
    print("\n" + "=" * 55)
    print(" STEP 2: Scanning raw data")
    print("=" * 55)

    all_images = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG", "*.JPEG"]:
        all_images.extend(RAW_DIR.rglob(ext))

    if not all_images:
        print(f"  ❌ No images found under {RAW_DIR}")
        return []

    print(f"  ✅ Found {len(all_images)} images total\n")

    # Show every unique leaf folder and its mapping
    folder_counts = {}
    for img in all_images:
        folder = img.parent.name
        folder_counts[folder] = folder_counts.get(folder, 0) + 1

    print(f"  {'Source folder':<40} {'Count':>6}   Maps to")
    print("  " + "-" * 65)
    for folder, count in sorted(folder_counts.items()):
        cls_id  = get_class_id(folder)
        mapping = f"[{cls_id}] {CLASS_NAMES[cls_id]}" if cls_id is not None else "⚠️  SKIPPED (unmapped)"
        print(f"  {folder:<40} {count:>6}   → {mapping}")

    return all_images


# ─────────────────────────────────────────────
def convert_and_split(all_images):
    print("\n" + "=" * 55)
    print(" STEP 3: Converting + Splitting 70/15/15")
    print("=" * 55)

    valid = [(img, get_class_id(img.parent.name))
             for img in all_images
             if get_class_id(img.parent.name) is not None]

    skipped = len(all_images) - len(valid)
    if skipped:
        print(f"  ⚠️  Skipped {skipped} images (unmapped folders)")
    print(f"  ✅ Using {len(valid)} images")

    if not valid:
        print("  ❌ Nothing to process.")
        return False

    random.seed(RANDOM_SEED)
    train, temp   = train_test_split(valid, test_size=0.30, random_state=RANDOM_SEED)
    val,   test   = train_test_split(temp,  test_size=0.50, random_state=RANDOM_SEED)

    print(f"  Train : {len(train)}")
    print(f"  Val   : {len(val)}")
    print(f"  Test  : {len(test)}")

    # Track used filenames to avoid collisions
    used_names = set()

    def copy_split(pairs, split_name):
        img_out = SPLITS_DIR / split_name / "images"
        lbl_out = SPLITS_DIR / split_name / "labels"
        for img_path, class_id in pairs:
            # Build unique filename: classname_originalname.ext
            base_name = f"{CLASS_NAMES[class_id]}_{img_path.name}"
            # Handle duplicates
            if base_name in used_names:
                base_name = f"{CLASS_NAMES[class_id]}_{img_path.parent.name}_{img_path.name}"
            used_names.add(base_name)

            # Copy image
            shutil.copy2(img_path, img_out / base_name)

            # Write YOLO label (full-image box — classification to detection standard)
            stem = Path(base_name).stem
            with open(lbl_out / f"{stem}.txt", "w") as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

    copy_split(train, "train")
    copy_split(val,   "val")
    copy_split(test,  "test")

    print("  ✅ Split complete!")
    return True


# ─────────────────────────────────────────────
def generate_yaml():
    print("\n" + "=" * 55)
    print(" STEP 4: Generating dataset.yaml")
    print("=" * 55)
    content  = "# Spinach Defect Detection\n"
    content += f"path: {SPLITS_DIR}\n"
    content += "train: train/images\n"
    content += "val: val/images\n"
    content += "test: test/images\n\n"
    content += f"nc: {len(CLASS_NAMES)}\n\nnames:\n"
    for i, name in enumerate(CLASS_NAMES):
        content += f"  {i}: {name}\n"
    yaml_path = BASE_DIR / "dataset.yaml"
    yaml_path.write_text(content)
    print(f"  ✅ Saved: {yaml_path}")


# ─────────────────────────────────────────────
def verify():
    print("\n" + "=" * 55)
    print(" STEP 5: Verification")
    print("=" * 55)
    all_ok = True
    for split in ["train", "val", "test"]:
        imgs = len(list((SPLITS_DIR / split / "images").glob("*.*")))
        lbls = len(list((SPLITS_DIR / split / "labels").glob("*.txt")))
        ok   = "✅" if imgs > 0 else "❌"
        print(f"  {ok} {split:<6}: {imgs} images | {lbls} labels")
        if imgs == 0:
            all_ok = False
    if all_ok:
        print(f"\n  🎉 Ready! Open spinach_detection.ipynb to train.")
        print(f"     dataset.yaml : {BASE_DIR / 'dataset.yaml'}")
    else:
        print("\n  ⚠️  Check messages above.")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  SPINACH DEFECT DETECTION — Dataset Preparation")
    print(f"  Running from: {BASE_DIR}")
    print("=" * 55)

    create_folders()
    images = scan_raw_data()
    if images:
        ok = convert_and_split(images)
        if ok:
            generate_yaml()
            verify()
    else:
        print("\n  ⏸  No images found. Check your data/raw/ folder.")
    print()
