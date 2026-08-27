import sys
from pathlib import Path
import tensorflow as tf

# Add ai folder to Python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from models.cnn import build_cnn
from data_loader import train_dataset, validation_dataset

# Automatically detect classes
NUM_CLASSES = len(train_dataset.class_names)

# Build CNN
model = build_cnn(NUM_CLASSES)

# Compile
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)
# Get final validation accuracy
final_val_accuracy = history.history["val_accuracy"][-1]

print(f"\nFinal Validation Accuracy: {final_val_accuracy * 100:.2f}%")

# Save model
SAVE_DIR = BASE_DIR.parent / "backend" / "saved_models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

model.save(SAVE_DIR / "cnn_model.keras")

print("\n✅ CNN Model Saved Successfully!")