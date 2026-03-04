from flask import Flask, request, send_file, make_response
from flask_cors import CORS
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from PIL import Image
import io
import os
import gc

# ==========================
# CPU + MEMORY CONTROL
# ==========================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

torch.set_grad_enabled(False)
torch.set_num_threads(1)

cv2.setNumThreads(0)

# ==========================
# APP INIT
# ==========================

app = Flask(__name__)
CORS(app)

YOLO_SIZE = 256
EFF_SIZE = 224
MAX_IMAGE_DIM = 640
MAX_TOMATOES = 8

# Upload protection (avoid RAM spike)
MAX_UPLOAD_SIZE = 5 * 1024 * 1024  # 5MB limit

CLASS_MAP = {0: "BAD", 1: "GOOD"}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_PATH = os.path.join(BASE_DIR, "model", "best.pt")
EFF_PATH = os.path.join(BASE_DIR, "model", "efficientnet_scripted.pt")

# ==========================
# LOAD MODELS
# ==========================

print("Loading YOLO...")
yolo = YOLO(YOLO_PATH)

# Reduce YOLO memory usage
yolo.model.fuse()

print("Loading EfficientNet...")
efficient_model = torch.jit.load(EFF_PATH, map_location="cpu")
efficient_model.eval()

print("Models Ready ✅")

# ==========================
# SMART RESIZE
# ==========================

def smart_resize(image):

    h, w = image.shape[:2]

    scale = MAX_IMAGE_DIM / max(h, w)

    if scale < 1:
        image = cv2.resize(image, (int(w*scale), int(h*scale)))

    return image

# ==========================
# ROUTES
# ==========================

@app.route("/")
def home():
    return "Tomato Sorting Backend Running"


@app.route("/detect", methods=["POST"])
def detect():

    try:

        if "image" not in request.files:
            return "No image uploaded", 400

        file = request.files["image"]

        # ==========================
        # FILE SIZE PROTECTION
        # ==========================

        file.seek(0, os.SEEK_END)
        file_length = file.tell()
        file.seek(0)

        if file_length > MAX_UPLOAD_SIZE:

            blank = np.zeros((400,600,3), dtype=np.uint8)

            cv2.putText(
                blank,
                "Image Too Large",
                (150,200),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                2
            )

            img = Image.fromarray(blank)

            img_io = io.BytesIO()
            img.save(img_io, format="JPEG")
            img_io.seek(0)

            response = make_response(send_file(img_io, mimetype="image/jpeg"))
            response.headers["X-Good-Tomatoes"] = "0"
            response.headers["X-Bad-Tomatoes"] = "0"

            return response

        file_bytes = file.read()

        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        image = np.array(image)

        image = smart_resize(image)

        original = image.copy()

        good_count = 0
        bad_count = 0

        # ==========================
        # YOLO DETECTION
        # ==========================

        with torch.inference_mode():

            results = yolo.predict(
                image,
                imgsz=YOLO_SIZE,
                conf=0.30,
                max_det=MAX_TOMATOES,
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

            count = 0

            for box in boxes:

                if count >= MAX_TOMATOES:
                    break

                x1, y1, x2, y2 = map(int, box)

                crop = image[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                crop = cv2.resize(crop, (EFF_SIZE, EFF_SIZE))

                crop = crop.astype("float32") / 255.0

                crop = np.transpose(crop, (2,0,1))

                tensor = torch.from_numpy(crop).unsqueeze(0)

                with torch.inference_mode():

                    output = efficient_model(tensor)

                    probs = torch.softmax(output, dim=1)

                    conf, pred = torch.max(probs, 1)

                pred = pred.item()
                conf = conf.item()

                # ==========================
                # COUNT GOOD / BAD
                # ==========================

                if pred == 1:
                    good_count += 1
                else:
                    bad_count += 1

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

                del tensor
                gc.collect()

                count += 1

        # ==========================
        # RETURN IMAGE + COUNTS
        # ==========================

        result = Image.fromarray(original)

        img_io = io.BytesIO()

        result.save(img_io, format="JPEG", quality=85)

        img_io.seek(0)

        gc.collect()

        response = make_response(send_file(img_io, mimetype="image/jpeg"))

        response.headers["X-Good-Tomatoes"] = str(good_count)
        response.headers["X-Bad-Tomatoes"] = str(bad_count)

        return response

    except Exception as e:

        print("ERROR:", e)

        blank = np.zeros((400,600,3), dtype=np.uint8)

        cv2.putText(
            blank,
            "Server Error",
            (160,200),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            2
        )

        img = Image.fromarray(blank)

        img_io = io.BytesIO()

        img.save(img_io, format="JPEG")

        img_io.seek(0)

        response = make_response(send_file(img_io, mimetype="image/jpeg"))

        response.headers["X-Good-Tomatoes"] = "0"
        response.headers["X-Bad-Tomatoes"] = "0"

        return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)