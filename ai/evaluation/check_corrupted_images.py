import os
from pathlib import Path
from PIL import Image

# Dataset path
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR.parent.parent / "dataset" / "raw"

corrupted_images = []

print("=" * 60)
print("CHECKING FOR CORRUPTED IMAGES")
print("=" * 60)

classes = sorted([
    folder for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(DATASET_PATH / folder)
])

for cls in classes:

    class_path = DATASET_PATH / cls

    for image_name in os.listdir(class_path):

        image_path = class_path / image_name

        try:
            img = Image.open(image_path)
            img.verify()

        except Exception:
            corrupted_images.append(image_path)

print("\nTotal Corrupted Images :", len(corrupted_images))

if corrupted_images:
    print("\nCorrupted Files:")
    for img in corrupted_images:
        print(img)
else:
    print("\n✅ No corrupted images found.")