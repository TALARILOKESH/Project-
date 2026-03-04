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
# CPU CONTROL
# ==============================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

torch.set_grad_enabled(False)
torch.set_num_threads(1)

cv2.setNumThreads(0)

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

print("Loading EfficientNet...")
efficient_model = torch.jit.load(EFF_PATH, map_location="cpu")
efficient_model.eval()

print("Models Loaded ✅")


# ==============================
# ROUTES
# ==============================
@app.route("/")
def home():
    return "Tomato Sorting Backend Running"


@app.route("/detect", methods=["POST"])
def detect():

    try:

        if "image" not in request.files:
            return "No image uploaded", 400

        file_bytes = request.files["image"].read()

        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        image = np.array(image)

        original = image.copy()

        # ==============================
        # YOLO DETECTION
        # ==============================
        with torch.inference_mode():

            results = yolo.predict(
                image,
                imgsz=IMAGE_SIZE_YOLO,
                conf=0.30,
                device="cpu",
                verbose=False
            )

        if len(results[0].boxes) == 0:

            cv2.putText(
                original,
                "No Tomato Detected",
                (50,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                2
            )

        else:

            boxes = results[0].boxes.xyxy.cpu().numpy()

            for box in boxes:

                x1, y1, x2, y2 = box.astype(int)

                crop = image[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                crop = cv2.resize(crop, (IMAGE_SIZE_EFF, IMAGE_SIZE_EFF))

                crop = crop.astype("float32") / 255.0

                crop = np.transpose(crop, (2,0,1))

                input_tensor = torch.from_numpy(crop).unsqueeze(0)

                # ==============================
                # CLASSIFICATION
                # ==============================
                with torch.inference_mode():

                    output = efficient_model(input_tensor)
                    probs = torch.softmax(output, dim=1)

                    confidence, predicted = torch.max(probs, dim=1)

                pred = predicted.item()
                prob = confidence.item()

                if pred == 1 and prob < GOOD_THRESHOLD:
                    pred = 0

                label = f"Tomato: {CLASS_MAP[pred]} ({prob:.2f})"

                color = (0,0,255) if pred == 0 else (0,255,0)

                # Draw box
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
        result = Image.fromarray(original)

        img_io = io.BytesIO()
        result.save(img_io, format="JPEG")
        img_io.seek(0)

        gc.collect()

        return send_file(img_io, mimetype="image/jpeg")

    except Exception as e:

        print("ERROR:", e)

        # Return error image instead of crashing
        blank = np.zeros((400,600,3), dtype=np.uint8)

        cv2.putText(
            blank,
            "Server Error",
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

        return send_file(img_io, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)