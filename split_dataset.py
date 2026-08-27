import os
import shutil
import random
from pathlib import Path

# =========================
# PATHS
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_DIR = PROJECT_ROOT / "dataset" / "raw"
TRAIN_DIR = PROJECT_ROOT / "dataset" / "train"
VAL_DIR = PROJECT_ROOT / "dataset" / "validation"
TEST_DIR = PROJECT_ROOT / "dataset" / "test"

# =========================
# SETTINGS
# =========================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

# =========================
# CREATE DIRECTORIES
# =========================

for folder in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# =========================
# PROCESS A-Z
# =========================

classes = sorted([
    folder.name
    for folder in RAW_DIR.iterdir()
    if folder.is_dir()
])

print("Classes found:", classes)
print("Total classes:", len(classes))

if len(classes) != 26:
    raise ValueError(
        f"Expected 26 classes (A-Z), but found {len(classes)} classes."
    )

for class_name in classes:

    source_dir = RAW_DIR / class_name

    train_class_dir = TRAIN_DIR / class_name
    val_class_dir = VAL_DIR / class_name
    test_class_dir = TEST_DIR / class_name

    train_class_dir.mkdir(parents=True, exist_ok=True)
    val_class_dir.mkdir(parents=True, exist_ok=True)
    test_class_dir.mkdir(parents=True, exist_ok=True)

    images = [
        file for file in source_dir.iterdir()
        if file.is_file()
    ]

    random.shuffle(images)

    total = len(images)

    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    train_images = images[:train_count]
    val_images = images[train_count:train_count + val_count]
    test_images = images[train_count + val_count:]

    print(
        f"{class_name}: "
        f"Train={len(train_images)}, "
        f"Validation={len(val_images)}, "
        f"Test={len(test_images)}"
    )

    # Copy files
    for image in train_images:
        shutil.copy2(image, train_class_dir / image.name)

    for image in val_images:
        shutil.copy2(image, val_class_dir / image.name)

    for image in test_images:
        shutil.copy2(image, test_class_dir / image.name)

print("\n================================")
print("Dataset splitting completed!")
print("================================")