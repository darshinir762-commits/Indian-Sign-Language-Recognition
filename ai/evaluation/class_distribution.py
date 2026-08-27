import os
from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR.parent.parent / "dataset" / "raw"

classes = []
counts = []

for folder in sorted(os.listdir(DATASET_PATH)):

    folder_path = DATASET_PATH / folder

    if folder_path.is_dir():

        total = len([
            img for img in os.listdir(folder_path)
            if img.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        classes.append(folder)
        counts.append(total)

plt.figure(figsize=(15,6))
plt.bar(classes, counts)

plt.title("Class Distribution")
plt.xlabel("Classes")
plt.ylabel("Number of Images")

plt.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.show()