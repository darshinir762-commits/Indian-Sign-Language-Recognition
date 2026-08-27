import sys
from pathlib import Path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=test_dataset.class_names
)

plt.figure(figsize=(8, 8))
disp.plot(cmap="Blues")
plt.title("EfficientNetB0 Confusion Matrix")

save_path = PROJECT_ROOT / "ai" / "reports" / "images" / "efficientnet_confusion_matrix.png"
plt.savefig(save_path)

plt.show()

print("✅ EfficientNetB0 Confusion Matrix Saved!")