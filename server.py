import io
import time
import base64
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from ultralytics import YOLO

# Category mappings based on VIT project proposal
# 0: Biodegradable, 1: Plastic, 2: Metal, 3: Hazardous
CATEGORY_MAP = {
    # 0: Biodegradable / Organic
    "banana": 0,
    "apple": 0,
    "sandwich": 0,
    "orange": 0,
    "broccoli": 0,
    "carrot": 0,
    "hot dog": 0,
    "pizza": 0,
    "donut": 0,
    "cake": 0,
    "potted plant": 0,
    # 1: Plastic / Paper / Packaging
    "bottle": 1,
    "cup": 1,
    "bowl": 1,
    "handbag": 1,
    "backpack": 1,
    "suitcase": 1,
    "frisbee": 1,
    "umbrella": 1,
    # 2: Metal / Cans / Hardware
    "fork": 2,
    "knife": 2,
    "spoon": 2,
    "scissors": 2,
    # 3: Hazardous / E-Waste
    "cell phone": 3,
    "mouse": 3,
    "remote": 3,
    "keyboard": 3,
    "laptop": 3,
    "microwave": 3,
    "toaster": 3,
    "clock": 3,
    "hair drier": 3,
    "toothbrush": 3,
}

CATEGORY_NAMES = ["Biodegradable", "Plastic", "Metal", "Hazardous"]
CATEGORY_COLORS = ["#22c55e", "#eab308", "#3b82f6", "#ef4444"]

app = FastAPI(
    title="Waste Sorter YOLOv11n Backend",
    description="Real-time YOLOv11n object detection service for automated waste sorting",
    version="1.0.0",
)

# Enable CORS for simulation frontend (local, localhost, and github.io)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path(__file__).resolve().parent / "yolo11n.pt"
print(f"Loading YOLO11n model from {MODEL_PATH}...")
model = YOLO(str(MODEL_PATH))
print(f"Loaded successfully. Classes: {len(model.names)}")


class Base64PredictRequest(BaseModel):
    image: str
    conf_threshold: Optional[float] = 0.15


def map_label_to_category(label: str) -> tuple[int, str]:
    lbl = label.lower().strip()
    if lbl in CATEGORY_MAP:
        cat_id = CATEGORY_MAP[lbl]
        return cat_id, CATEGORY_NAMES[cat_id]

    # Substring heuristic
    for key, cat_id in CATEGORY_MAP.items():
        if key in lbl or lbl in key:
            return cat_id, CATEGORY_NAMES[cat_id]

    # Default fallback to Plastic / General Dry Recyclable (1)
    return 1, CATEGORY_NAMES[1]


def run_inference(image: Image.Image, conf_threshold: float = 0.15):
    t_start = time.perf_counter()
    img_w, img_h = image.size

    # First pass with user or default threshold
    results = model.predict(source=image, conf=conf_threshold, imgsz=320, verbose=False)
    result = results[0]

    # If nothing detected and threshold was high, retry with lower threshold (0.08)
    if len(result.boxes) == 0 and conf_threshold > 0.08:
        results = model.predict(source=image, conf=0.08, imgsz=320, verbose=False)
        result = results[0]

    infer_time_ms = round((time.perf_counter() - t_start) * 1000, 1)
    detections = []

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        label = model.names[cls_id]
        conf = float(box.conf[0].item())
        xyxy = [float(v) for v in box.xyxy[0].tolist()]

        cat_id, cat_name = map_label_to_category(label)

        detections.append({
            "label": label,
            "category_id": cat_id,
            "category_name": cat_name,
            "color": CATEGORY_COLORS[cat_id],
            "confidence": round(conf * 100, 1),
            "bbox": {
                "x1": round(xyxy[0], 1),
                "y1": round(xyxy[1], 1),
                "x2": round(xyxy[2], 1),
                "y2": round(xyxy[3], 1),
                "width": round(xyxy[2] - xyxy[0], 1),
                "height": round(xyxy[3] - xyxy[1], 1),
                "normalized": [
                    round(xyxy[0] / img_w, 4),
                    round(xyxy[1] / img_h, 4),
                    round(xyxy[2] / img_w, 4),
                    round(xyxy[3] / img_h, 4),
                ],
            },
            "is_fallback": False,
        })

    detections.sort(key=lambda d: d["confidence"], reverse=True)

    # If still no bounding box was detected by standard COCO classes, provide an estimated detection
    # so the automated sorting chute and physical sensors always take action on deposited waste
    if not detections:
        detections.append({
            "label": "unclassified_waste",
            "category_id": 1,
            "category_name": CATEGORY_NAMES[1],
            "color": CATEGORY_COLORS[1],
            "confidence": 72.0,
            "bbox": {
                "x1": round(img_w * 0.15, 1),
                "y1": round(img_h * 0.15, 1),
                "x2": round(img_w * 0.85, 1),
                "y2": round(img_h * 0.85, 1),
                "width": round(img_w * 0.70, 1),
                "height": round(img_h * 0.70, 1),
                "normalized": [0.15, 0.15, 0.85, 0.85],
            },
            "is_fallback": True,
        })

    primary = detections[0]

    return {
        "status": "success",
        "inference_time_ms": infer_time_ms,
        "speed": result.speed,
        "detection_count": len(detections),
        "primary_detection": primary,
        "detections": detections,
    }


@app.get("/")
def root():
    return {
        "service": "YOLOv11n Waste Management Backend",
        "status": "online",
        "model": "yolo11n.pt",
        "classes_count": len(model.names),
        "categories": [
            {"id": i, "name": CATEGORY_NAMES[i], "color": CATEGORY_COLORS[i]}
            for i in range(4)
        ],
    }


@app.post("/predict")
async def predict_upload(file: UploadFile = File(...), conf_threshold: float = 0.15):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    return run_inference(image, conf_threshold)


@app.post("/predict-base64")
async def predict_base64(payload: Base64PredictRequest):
    try:
        data = payload.image
        if "," in data:
            data = data.split(",", 1)[1]
        decoded = base64.b64decode(data)
        image = Image.open(io.BytesIO(decoded)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode base64 image: {e}")

    return run_inference(image, payload.conf_threshold or 0.15)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
