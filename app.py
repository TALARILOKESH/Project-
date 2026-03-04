from flask import Flask, request, send_file
from flask_cors import CORS
from ultralytics import YOLO
import torch
import cv2
import numpy as np
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

print("Models Loaded")

# ==============================
# ROUTES
# ==============================
@app.route("/")
def home():
    return "Tomato Sorting Backend Running"


@app.route("/detect", methods=["POST"])
def detect():

    if "image" not in request.files:
        return "No image uploaded", 400

    file_bytes = request.files["image"].read()
    np_img = np.frombuffer(file_bytes, np.uint8)

    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ==============================
    # SMART RESIZE
    # ==============================
    h, w = image.shape[:2]

    MAX_DIM = 640
    scale = MAX_DIM / max(h, w)

    if scale < 1:
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    original = image.copy()

    # ==============================
    # YOLO DETECTION (Improved NMS)
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

    boxes = results[0].boxes.xyxy.numpy()

    del results
    gc.collect()

    crops = []
    coords = []

    H, W = image.shape[:2]

    # ==============================
    # PREPARE CROPS
    # ==============================
    for b in boxes:

        x1, y1, x2, y2 = b.astype(int)

        pad = int(0.05 * max(x2-x1, y2-y1))

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

        # EfficientNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        crop = (crop - mean) / std

        crop = np.transpose(crop, (2,0,1))

        crops.append(crop)
        coords.append((x1,y1,x2,y2))

    if len(crops) == 0:
        return "No valid crop"

    preds = []
    probs = []

    BATCH = 4

    # ==============================
    # CLASSIFICATION + TTA
    # ==============================
    for i in range(0,len(crops),BATCH):

        batch = torch.tensor(crops[i:i+BATCH])

        batch_flip_h = torch.flip(batch, dims=[3])
        batch_flip_v = torch.flip(batch, dims=[2])

        with torch.inference_mode():

            out1 = efficient_model(batch)
            out2 = efficient_model(batch_flip_h)
            out3 = efficient_model(batch_flip_v)

            p = (
                torch.softmax(out1, dim=1) +
                torch.softmax(out2, dim=1) +
                torch.softmax(out3, dim=1)
            ) / 3

        pr = torch.argmax(p, dim=1)

        preds.extend(pr.tolist())
        probs.extend(p.tolist())

        del batch
        del out1
        del out2
        del out3
        gc.collect()

    # ==============================
    # DRAW RESULTS
    # ==============================
    for i,(x1,y1,x2,y2) in enumerate(coords):

        pred = preds[i]
        prob = probs[i][pred]

        if pred == 1 and prob < GOOD_THRESHOLD:
            pred = 0

        label = f"Tomato: {CLASS_MAP[pred]} ({prob:.2f})"

        color = (0,0,255) if pred == 0 else (0,255,0)

        cv2.rectangle(original,(x1,y1),(x2,y2),color,2)

        cv2.putText(
            original,
            label,
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # ==============================
    # RETURN IMAGE
    # ==============================
    success, buffer = cv2.imencode(
    ".jpg",
    cv2.cvtColor(original, cv2.COLOR_RGB2BGR),
    [int(cv2.IMWRITE_JPEG_QUALITY), 80]
)

if not success:
    return "Image encoding failed", 500

return send_file(
    io.BytesIO(buffer.tobytes()),
    mimetype="image/jpeg",
    as_attachment=False,
    download_name="result.jpg"
)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)