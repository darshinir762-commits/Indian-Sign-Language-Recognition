import os
from pathlib import Path
from PIL import Image
import numpy as np

# Dataset path
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR.parent.parent / "dataset" / "raw"

widths = []
heights = []
channels = []

for folder in sorted(os.listdir(DATASET_PATH)):

    folder_path = DATASET_PATH / folder

    if not folder_path.is_dir():
        continue

    for image_name in os.listdir(folder_path):

        image_path = folder_path / image_name

        try:
            img = Image.open(image_path)

            widths.append(img.width)
            heights.append(img.height)

            img_array = np.array(img)

            if len(img_array.shape) == 3:
                channels.append(img_array.shape[2])
            else:
                channels.append(1)

        except Exception:
            continue

print("=" * 60)
print("IMAGE STATISTICS")
print("=" * 60)

print(f"Total Images      : {len(widths)}")
print(f"Average Width     : {sum(widths)/len(widths):.2f}")
print(f"Average Height    : {sum(heights)/len(heights):.2f}")
print(f"Minimum Width     : {min(widths)}")
print(f"Maximum Width     : {max(widths)}")
print(f"Minimum Height    : {min(heights)}")
print(f"Maximum Height    : {max(heights)}")
print(f"Unique Channels   : {set(channels)}")