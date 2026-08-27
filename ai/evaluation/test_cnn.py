import sys
from pathlib import Path
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from training.data_loader import test_dataset

MODEL_PATH = BASE_DIR.parent / "backend" / "saved_models" / "cnn_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

loss, accuracy = model.evaluate(test_dataset)

print("\n==========================")
print("CNN TEST RESULTS")
print("==========================")
print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy:.4f}")