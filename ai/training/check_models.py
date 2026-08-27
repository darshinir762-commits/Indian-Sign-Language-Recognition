from pathlib import Path
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent.parent
SAVE_DIR = BASE_DIR.parent / "backend" / "saved_models"

models = {
    "CNN": "cnn_model.keras",
    "MobileNetV2": "mobilenet_model.keras",
    "EfficientNetB0": "efficientnet_model.keras",
}

for name, filename in models.items():

    path = SAVE_DIR / filename

    print("\n" + "=" * 50)
    print(name)

    if not path.exists():
        print("❌ Model not found:", path)
        continue

    model = tf.keras.models.load_model(path)

    print("Input shape :", model.input_shape)
    print("Output shape:", model.output_shape)