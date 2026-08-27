import sys
from pathlib import Path
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from ai.training.data_loader import test_dataset

MODEL_PATH = PROJECT_ROOT / "backend" / "saved_models" / "mobilenet_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

loss, accuracy = model.evaluate(test_dataset)

print("\n==============================")
print("MOBILENETV2 TEST RESULTS")
print("==============================")
print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy:.4f}")