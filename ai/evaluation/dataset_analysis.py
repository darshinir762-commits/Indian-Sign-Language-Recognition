import os
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR.parent.parent / "dataset" / "raw"

classes = []
image_counts = []

print("=" * 60)
print("INDIAN SIGN LANGUAGE DATASET ANALYSIS")
print("=" * 60)

if not DATASET_PATH.exists():
    print("Dataset not found!")
    exit()

folders = sorted(
    [f for f in os.listdir(DATASET_PATH)
     if os.path.isdir(DATASET_PATH / f)]
)

total_images = 0

for folder in folders:

    folder_path = DATASET_PATH / folder

    images = [
        img for img in os.listdir(folder_path)
        if img.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    count = len(images)

    classes.append(folder)
    image_counts.append(count)

    total_images += count

    print(f"{folder} : {count}")

print("\nTotal Classes :", len(classes))
print("Total Images :", total_images)

plt.figure(figsize=(12,5))
plt.bar(classes, image_counts)

plt.title("Class Distribution")
plt.xlabel("Classes")
plt.ylabel("Images")

plt.tight_layout()
plt.show()