import sys
from pathlib import Path
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from training.data_loader import test_dataset

MODEL_PATH = BASE_DIR.parent / "backend" / "saved_models" / "cnn_model.keras"

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

# Save report
report_path = BASE_DIR / "reports" / "metrics" / "classification_report.txt"

with open(report_path, "w") as f:
    f.write(report)

print("\n✅ Classification report saved.")