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

# =====================================
# CPU + MEMORY CONTROL (Render Safe)
# =====================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

torch.set_grad_enabled(False)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

cv2.setNumThreads(0)

# =====================================
# APP INIT
# =====================================

app = Flask(__name__)
CORS(app)

YOLO_SIZE = 256
EFF_SIZE = 224

MAX_IMAGE_DIM = 640

CLASS_MAP = {0: "BAD", 1: "GOOD"}

# =====================================
# PATHS
# =====================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_PATH = os.path.join(BASE_DIR, "model", "best.pt")
EFF_PATH = os.path.join(BASE_DIR, "model", "efficientnet_scripted.pt")

# =====================================
# LOAD MODELS ONCE
# =====================================

print("Loading YOLO...")
yolo = YOLO(YOLO_PATH)

print("Loading EfficientNet...")
efficient_model = torch.jit.load(EFF_PATH, map_location="cpu")
efficient_model.eval()

print("Models Loaded Successfully ✅")


# =====================================
# SMART RESIZE FUNCTION
# =====================================

def smart_resize(image):

    h, w = image.shape[:2]

    scale = MAX_IMAGE_DIM / max(h, w)

    if scale < 1:
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    return image


# =====================================
# ROUTES
# =====================================

@app.route("/")
def home():
    return "Tomato Sorting Backend Running"


@app.route("/detect", methods=["POST"])
def detect():

    try:

        if "image" not in request.files:
            return "No image uploaded", 400

        # =====================================
        # LOAD IMAGE
        # =====================================

        file_bytes = request.files["image"].read()

        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        image = np.array(image)

        image = smart_resize(image)

        original = image.copy()

        # =====================================
        # YOLO DETECTION
        # =====================================

        with torch.inference_mode():

            results = yolo.predict(
                image,
                imgsz=YOLO_SIZE,
                conf=0.30,
                device="cpu",
                verbose=False
            )

        if not results or len(results[0].boxes) == 0:

            cv2.putText(
                original,
                "No Tomato Detected",
                (40,60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                2
            )

        else:

            boxes = results[0].boxes.xyxy.cpu().numpy()

            crops = []
            coords = []

            # =====================================
            # CROP TOMATOES
            # =====================================

            for box in boxes:

                x1, y1, x2, y2 = map(int, box)

                crop = image[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                crop = cv2.resize(crop, (EFF_SIZE, EFF_SIZE))

                crop = crop.astype("float32") / 255.0

                crop = np.transpose(crop, (2,0,1))

                crops.append(crop)
                coords.append((x1,y1,x2,y2))

            if len(crops) > 0:

                batch = torch.tensor(crops)

                # =====================================
                # BATCH CLASSIFICATION (FAST)
                # =====================================

                with torch.inference_mode():

                    outputs = efficient_model(batch)

                    probs = torch.softmax(outputs, dim=1)

                    confs, preds = torch.max(probs, 1)

                preds = preds.tolist()
                confs = confs.tolist()

                # =====================================
                # DRAW BOXES
                # =====================================

                for i,(x1,y1,x2,y2) in enumerate(coords):

                    pred = preds[i]
                    conf = confs[i]

                    label = f"Tomato: {CLASS_MAP[pred]} ({conf:.2f})"

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

                del batch

        # =====================================
        # MEMORY CLEANUP
        # =====================================

        gc.collect()

        # =====================================
        # RETURN IMAGE
        # =====================================

        result = Image.fromarray(original)

        img_io = io.BytesIO()

        result.save(img_io, format="JPEG", quality=85)

        img_io.seek(0)

        return send_file(img_io, mimetype="image/jpeg")

    except Exception as e:

        print("ERROR:", e)

        error_img = np.zeros((400,600,3), dtype=np.uint8)

        cv2.putText(
            error_img,
            "Server Error",
            (160,200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )

        img = Image.fromarray(error_img)

        img_io = io.BytesIO()

        img.save(img_io, format="JPEG")

        img_io.seek(0)

        return send_file(img_io, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)