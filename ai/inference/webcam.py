import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

MODEL_PATH = PROJECT_ROOT / "backend" / "saved_models" / "efficientnet_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ["A", "B", "C", "D"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    roi = frame[100:324, 100:324]

    img = cv2.resize(roi, (224, 224))
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)

    cv2.rectangle(frame, (100, 100), (324, 324), (0, 255, 0), 2)

    cv2.putText(
        frame,
        f"{predicted_class} ({confidence*100:.2f}%)",
        (100, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Indian Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()