import sys
from pathlib import Path
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from ai.training.data_loader import test_dataset

MODEL_PATH = PROJECT_ROOT / "backend" / "saved_models" / "efficientnet_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

y_true = []
y_pred = []

for images, labels in test_dataset:
    predictions = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(predictions, axis=1))

report = classification_report(
    y_true,
    y_pred,
    target_names=test_dataset.class_names
)

print(report)

save_path = PROJECT_ROOT / "ai" / "reports" / "metrics" / "efficientnet_classification_report.txt"

with open(save_path, "w") as f:
    f.write(report)

print("✅ EfficientNetB0 Classification Report Saved!")