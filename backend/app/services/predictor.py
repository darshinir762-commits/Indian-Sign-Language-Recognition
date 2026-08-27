from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
import time

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_DIR = BASE_DIR / "saved_models"

MODEL_PATHS = {
    "cnn": MODEL_DIR / "cnn_model.keras",
    "mobilenet": MODEL_DIR / "mobilenet_model.keras",
    "efficientnet": MODEL_DIR / "efficientnet_model.keras",
}

CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F",
    "G", "H", "I", "J", "K", "L",
    "M", "N", "O", "P", "Q", "R",
    "S", "T", "U", "V", "W", "X",
    "Y", "Z"
]


# Load models only when needed
models = {}


def get_model(model_name):
    if model_name not in MODEL_PATHS:
        raise ValueError(
            f"Invalid model: {model_name}. "
            f"Choose from: {list(MODEL_PATHS.keys())}"
        )

    if model_name not in models:
        model_path = MODEL_PATHS[model_name]

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        print(f"Loading {model_name} model...")

        models[model_name] = tf.keras.models.load_model(model_path)

        print(f"{model_name} model loaded successfully!")

    return models[model_name]


def predict(image_path, model_name="efficientnet"):

    model = get_model(model_name)

    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))

    img = np.array(img)
    img = np.expand_dims(img, axis=0)

    start_time = time.time()

    prediction = model(img, training=False).numpy()
    print("RAW PREDICTION:", prediction)
    print("OUTPUT SHAPE:", prediction.shape)
    print("OUTPUT SUM:", np.sum(prediction))
    print("MAX VALUE:", np.max(prediction))

    inference_time = time.time() - start_time

    label = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    # Get Top-3 predictions
    top_3_indices = np.argsort(prediction[0])[-3:][::-1]

    top_3 = [
      {
        "label": CLASS_NAMES[i],
        "confidence": round(float(prediction[0][i]) * 100, 2)
       }
        for i in top_3_indices
]

    return label, confidence, inference_time, top_3