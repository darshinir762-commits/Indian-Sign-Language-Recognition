import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_PATH = BASE_DIR / "dataset" / "raw"

TRAIN_PATH = BASE_DIR / "dataset" / "train"
VAL_PATH = BASE_DIR / "dataset" / "validation"
TEST_PATH = BASE_DIR / "dataset" / "test"

# Create output folders
for path in [TRAIN_PATH, VAL_PATH, TEST_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# Process each class
for class_name in sorted(os.listdir(RAW_PATH)):

    class_folder = RAW_PATH / class_name

    if not class_folder.is_dir():
        continue

    images = [
        img for img in os.listdir(class_folder)
        if img.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Train 70%, Temp 30%
    train_imgs, temp_imgs = train_test_split(
        images,
        test_size=0.30,
        random_state=42,
        shuffle=True
    )

    # Validation 15%, Test 15%
    val_imgs, test_imgs = train_test_split(
        temp_imgs,
        test_size=0.50,
        random_state=42,
        shuffle=True
    )

    # Create class folders
    for folder in [TRAIN_PATH, VAL_PATH, TEST_PATH]:
        (folder / class_name).mkdir(parents=True, exist_ok=True)

    # Copy files
    def copy_images(image_list, destination):
        for img in image_list:
            shutil.copy2(
                class_folder / img,
                destination / class_name / img
            )

    copy_images(train_imgs, TRAIN_PATH)
    copy_images(val_imgs, VAL_PATH)
    copy_images(test_imgs, TEST_PATH)

    print(
        f"{class_name} -> "
        f"Train:{len(train_imgs)} "
        f"Val:{len(val_imgs)} "
        f"Test:{len(test_imgs)}"
    )

print("\n✅ Dataset split completed successfully!")