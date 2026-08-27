import sys
from pathlib import Path
import tensorflow as tf
import matplotlib.pyplot as plt

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import Model and Dataset
from ai.models.efficientnet import build_efficientnet
from ai.training.data_loader import train_dataset, validation_dataset

# Number of Classes
NUM_CLASSES = len(train_dataset.class_names)
print("Classes:", train_dataset.class_names)
print("Number of classes:", NUM_CLASSES)

# Build Model
model = build_efficientnet(NUM_CLASSES)

# Compile Model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Train Model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)
# Get final validation accuracy
final_val_accuracy = history.history["val_accuracy"][-1]

print(f"\nFinal Validation Accuracy: {final_val_accuracy * 100:.2f}%")

# Save Model
SAVE_DIR = PROJECT_ROOT / "backend" / "saved_models"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

model.save(SAVE_DIR / "efficientnet_model.keras")

print("\n✅ EfficientNetB0 Model Saved Successfully!")

# Create Reports Folder
REPORTS_DIR = PROJECT_ROOT / "reports" / "images"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================
# Accuracy Graph
# ==========================
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("EfficientNetB0 Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.savefig(REPORTS_DIR / "efficientnet_accuracy.png")
plt.show()

# ==========================
# Loss Graph
# ==========================
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("EfficientNetB0 Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.savefig(REPORTS_DIR / "efficientnet_loss.png")
plt.show()

print("✅ Accuracy and Loss graphs saved successfully!")