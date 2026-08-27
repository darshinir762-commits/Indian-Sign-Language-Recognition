import sys
from pathlib import Path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

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

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=test_dataset.class_names
)

plt.figure(figsize=(8,8))
disp.plot(cmap="Blues")
plt.title("CNN Confusion Matrix")
plt.savefig("../../ai/reports/images/confusion_matrix.png")
plt.show()

print("✅ Confusion Matrix Saved")