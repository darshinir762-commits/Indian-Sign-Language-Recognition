from pathlib import Path
from PIL import Image

dataset_path = Path("../../dataset")

folders = ["train", "validation", "test"]

bad_images = []

for folder in folders:
    print(f"\nChecking {folder}...")

    for image_path in (dataset_path / folder).rglob("*.jpg"):
        try:
            with Image.open(image_path) as img:
                img.verify()

            # Check if file size is zero
            if image_path.stat().st_size == 0:
                bad_images.append(image_path)

        except Exception:
            bad_images.append(image_path)

print("\n==============================")

if len(bad_images) == 0:
    print("✅ No bad images found.")
else:
    print(f"❌ Found {len(bad_images)} bad image(s):\n")

    for img in bad_images:
        print(img)