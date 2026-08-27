import sys
from pathlib import Path
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

BASE_DIR = PROJECT_ROOT / "ai"

from ai.models.mobilenet import build_mobilenet
from ai.training.data_loader import train_dataset, validation_dataset
NUM_CLASSES = len(train_dataset.class_names)

model = build_mobilenet(NUM_CLASSES)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)
# Get final validation accuracy
final_val_accuracy = history.history["val_accuracy"][-1]

print(f"\nFinal Validation Accuracy: {final_val_accuracy * 100:.2f}%")

SAVE_DIR = BASE_DIR.parent / "backend" / "saved_models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

model.save(SAVE_DIR / "mobilenet_model.keras")

print("✅ MobileNetV2 Model Saved Successfully!")