import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

# Dataset path
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR.parent.parent / "dataset" / "raw"

# Get all class folders
classes = sorted([
    folder for folder in os.listdir(DATASET_PATH)
    if os.path.isdir(DATASET_PATH / folder)
])

plt.figure(figsize=(12,8))

# Show one random image from first 9 classes
for i, cls in enumerate(classes[:9]):

    class_path = DATASET_PATH / cls

    images = [
        img for img in os.listdir(class_path)
        if img.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if len(images) == 0:
        continue

    image = random.choice(images)

    img = Image.open(class_path / image)

    plt.subplot(3,3,i+1)
    plt.imshow(img)
    plt.title(cls)
    plt.axis("off")

plt.tight_layout()
plt.show()