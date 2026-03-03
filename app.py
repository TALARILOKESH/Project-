from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from PIL import Image
import io
import os
import gc

app = Flask(__name__)
CORS(app)

# ----------------------------
# CONFIG
# ----------------------------
IMAGE_SIZE_YOLO = 256   # reduced for lower RAM
IMAGE_SIZE_EFF = 224
CLASS_NAMES = ["Bad Tomato", "Good Tomato"]

# ----------------------------
# PATHS
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_PATH = os.path.join(BASE_DIR, "model", "best.pt")
EFF_PATH = os.path.join(BASE_DIR, "model", "efficientnet_scripted.pt")

# ----------------------------
# LOAD MODELS (ONCE)
# ----------------------------
print("Loading YOLO model...")
yolo_model = YOLO(YOLO_PATH)
yolo_model.fuse()   # reduce memory slightly

print("Loading TorchScript EfficientNet...")
efficient_model = torch.jit.load(EFF_PATH, map_location="cpu")
efficient_model.eval()

print("Models Loaded Successfully ✅")

# ----------------------------
# ROOT ROUTE
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    return "Tomato Sorting Backend Running ✅"

# ----------------------------
# DETECT ROUTE
# ----------------------------
@app.route("/detect", methods=["POST"])
def detect():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        file_bytes = request.files["image"].read()
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        image_np = np.array(image)

        # Resize BEFORE YOLO
        image_np = cv2.resize(image_np, (IMAGE_SIZE_YOLO, IMAGE_SIZE_YOLO))

        # YOLO Detection
        results = yolo_model(image_np, imgsz=IMAGE_SIZE_YOLO, verbose=False)

        if len(results[0].boxes) == 0:
            return jsonify({"result": "No Tomato Detected"})

        box = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box

        cropped = image_np[y1:y2, x1:x2]

        if cropped.size == 0:
            return jsonify({"result": "Detection Error"})

        # EfficientNet Classification
        cropped = cv2.resize(cropped, (IMAGE_SIZE_EFF, IMAGE_SIZE_EFF))
        cropped = cropped.astype("float32") / 255.0
        cropped = np.transpose(cropped, (2, 0, 1))
        input_tensor = torch.tensor(cropped).unsqueeze(0)

        with torch.inference_mode():
            output = efficient_model(input_tensor)
            predicted = torch.argmax(output, dim=1).item()

        gc.collect()  # free memory

        return jsonify({"result": CLASS_NAMES[predicted]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)