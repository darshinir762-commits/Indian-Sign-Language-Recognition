import os
from pathlib import Path
from PIL import Image

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

TRAIN_PATH = BASE_DIR / "dataset" / "train"
OUTPUT_PATH = BASE_DIR / "dataset" / "processed"

IMAGE_SIZE = (224, 224)

print("=" * 60)
print("IMAGE PREPROCESSING")
print("=" * 60)

OUTPUT_PATH.mkdir(exist_ok=True)

total = 0

for class_name in sorted(os.listdir(TRAIN_PATH)):

    class_path = TRAIN_PATH / class_name

    if not class_path.is_dir():
        continue

    output_class = OUTPUT_PATH / class_name
    output_class.mkdir(exist_ok=True)

    for image_name in os.listdir(class_path):

        image_path = class_path / image_name

        try:

            img = Image.open(image_path)

            img = img.resize(IMAGE_SIZE)

            img.save(output_class / image_name)

            total += 1

        except Exception:
            continue

print(f"\nProcessed Images : {total}")
print("\n✅ Image resizing completed.")