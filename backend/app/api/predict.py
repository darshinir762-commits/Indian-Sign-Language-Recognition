from fastapi import APIRouter, UploadFile, File, Form
import shutil
from pathlib import Path
import time

from app.services.predictor import predict

router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("/predict")
async def predict_image(
    file: UploadFile = File(...),
    model: str = Form("efficientnet")
):
    image_path = UPLOAD_DIR / file.filename

    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    start_time = time.perf_counter()

    label, confidence, _, top_3 = predict(image_path, model)

    inference_time_ms = (time.perf_counter() - start_time) * 1000

    print("🔥 ACTUAL INFERENCE TIME:", inference_time_ms, "ms")
    return {
    "prediction": label,
    "confidence": round(confidence * 100, 2),
    "model": model,
    "inference_time": inference_time_ms,
    "top_3": top_3
}