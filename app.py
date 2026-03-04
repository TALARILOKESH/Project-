from flask import Flask, request, send_file
from flask_cors import CORS
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from PIL import Image
import io
import os
import gc

# ==============================
# CPU + MEMORY CONTROL
# ==============================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

torch.set_grad_enabled(False)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

cv2.setNumThreads(0)
cv2.setUseOptimized(True)

# ==============================
# APP INIT
# ==============================
app = Flask(__name__)
CORS(app)

IMAGE_SIZE_YOLO = 192
IMAGE_SIZE_EFF = 224

GOOD_THRESHOLD = 0.75

CLASS_MAP = {0: "BAD", 1: "GOOD"}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_PATH = os.path.join(BASE_DIR, "model", "best.pt")
EFF_PATH = os.path.join(BASE_DIR, "model", "efficientnet_scripted.pt")

# ==============================
# LOAD MODELS
# ==============================
print("Loading YOLO...")
yolo = YOLO(YOLO_PATH)
yolo.model.eval()

print("Loading EfficientNet...")
efficient_model = torch.jit.load(EFF_PATH, map_location="cpu")
efficient_model.eval()

print("Models Loaded Successfully ✅")

# ==============================
# ROUTES
# ==============================
@app.route("/")
def home():
    return "Tomato Sorting Backend Running ✅"


@app.route("/detect", methods=["POST"])
def detect():

    if "image" not in request.files:
        return "No image uploaded", 400

    # ==============================
    # LOAD IMAGE
    # ==============================
    file_bytes = request.files["image"].read()

    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    image = np.array(image)

    original = image.copy()

    # ==============================
    # SMART RESIZE
    # ==============================
    h, w = image.shape[:2]

    MAX_DIM = 640
    scale = MAX_DIM / max(h, w)

    if scale < 1:
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    # ==============================
    # YOLO DETECTION
    # ==============================
    with torch.inference_mode():

        results = yolo.predict(
            image,
            imgsz=IMAGE_SIZE_YOLO,
            conf=0.30,
            iou=0.50,
            max_det=20,
            device="cpu",
            verbose=False
        )

    if len(results[0].boxes) == 0:
        return "No Tomato Detected"

    boxes = results[0].boxes.xyxy.cpu().numpy()

    del results
    gc.collect()

    H, W = image.shape[:2]

    # ==============================
    # PROCESS EACH DETECTION
    # ==============================
    for b in boxes:

        x1, y1, x2, y2 = b.astype(int)

        pad = int(0.05 * max(x2 - x1, y2 - y1))

        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(W, x2 + pad)
        y2 = min(H, y2 + pad)

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        # center crop improvement
        h2, w2 = crop.shape[:2]
        margin = int(0.1 * min(h2, w2))

        crop = crop[margin:h2-margin, margin:w2-margin]

        crop = cv2.resize(crop, (IMAGE_SIZE_EFF, IMAGE_SIZE_EFF))

        crop = crop.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        crop = (crop - mean) / std

        crop = np.transpose(crop, (2, 0, 1))
        input_tensor = torch.tensor(crop).unsqueeze(0)

        # ==============================
        # CLASSIFICATION
        # ==============================
        with torch.inference_mode():

            out1 = efficient_model(input_tensor)
            out2 = efficient_model(torch.flip(input_tensor, dims=[3]))
            out3 = efficient_model(torch.flip(input_tensor, dims=[2]))

            probs = (
                torch.softmax(out1, dim=1) +
                torch.softmax(out2, dim=1) +
                torch.softmax(out3, dim=1)
            ) / 3

        pred = torch.argmax(probs, dim=1).item()
        prob = probs[0][pred].item()

        if pred == 1 and prob < GOOD_THRESHOLD:
            pred = 0

        label = f"Tomato: {CLASS_MAP[pred]} ({prob:.2f})"

        color = (0, 0, 255) if pred == 0 else (0, 255, 0)

        # ==============================
        # DRAW BOUNDING BOX
        # ==============================
        cv2.rectangle(original, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            original,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        del input_tensor
        gc.collect()

    # ==============================
    # RETURN IMAGE
    # ==============================
    result_image = Image.fromarray(original)

    img_io = io.BytesIO()
    result_image.save(img_io, format="JPEG")
    img_io.seek(0)

    gc.collect()

    return send_file(img_io, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)