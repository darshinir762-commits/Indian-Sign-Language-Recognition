import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from ai.inference.predictor import predict_image

IMAGE_PATH = PROJECT_ROOT / "dataset" / "test" / "A" / "Image_1679027620.877586.jpg"

prediction, confidence = predict_image(IMAGE_PATH)

print("=" * 40)
print("Prediction :", prediction)
print("Confidence :", round(confidence * 100, 2), "%")