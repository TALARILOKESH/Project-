from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)

# ----------------------------
# CONFIG
# ----------------------------
IMAGE_SIZE_YOLO = 320
IMAGE_SIZE_EFF = 224
CLASS_NAMES = ["Bad Tomato", "Good Tomato"]

# ----------------------------
# PATH SETUP
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO_PATH = os.path.join(BASE_DIR, "model", "best.pt")
EFF_PATH = os.path.join(BASE_DIR, "model", "efficientnet_b0_best.pth")

# ----------------------------
# LOAD YOLO MODEL
# ----------------------------
print("Loading YOLO model...")
yolo_model = YOLO(YOLO_PATH)

# ----------------------------
# LOAD EFFICIENTNET MODEL (state_dict)
# ----------------------------
print("Loading EfficientNet model...")

device = torch.device("cpu")

# Recreate EfficientNet-B0 architecture
efficient_model = models.efficientnet_b0(weights=None)

# Modify classifier to match your training (2 classes)
efficient_model.classifier[1] = nn.Linear(
    efficient_model.classifier[1].in_features,
    2
)

# Load state_dict weights
efficient_model.load_state_dict(
    torch.load(EFF_PATH, map_location=device)
)

efficient_model.eval()

print("Models Loaded Successfully ✅")

# ----------------------------
# ROUTE
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    return "Tomato Sorting Backend Running ✅"
@app.route("/test", methods=["GET"])
def test():
    return "TEST ROUTE WORKING"
@app.route("/detect", methods=["POST"])
def detect():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        # Read image
        file_bytes = request.files["image"].read()
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        image_np = np.array(image)

        # Resize before YOLO (IMPORTANT for speed)
        image_np = cv2.resize(image_np, (IMAGE_SIZE_YOLO, IMAGE_SIZE_YOLO))

        # ---------------- YOLO DETECTION ----------------
        results = yolo_model(
            image_np,
            imgsz=IMAGE_SIZE_YOLO,
            conf=0.4,
            verbose=False
        )

        if len(results[0].boxes) == 0:
            return jsonify({"result": "No Tomato Detected"})

        # Take first detected box
        box = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box

        cropped = image_np[y1:y2, x1:x2]

        if cropped.size == 0:
            return jsonify({"result": "Detection Error"})

        # ---------------- EfficientNet Classification ----------------
        cropped = cv2.resize(cropped, (IMAGE_SIZE_EFF, IMAGE_SIZE_EFF))
        cropped = cropped.astype("float32") / 255.0
        cropped = np.transpose(cropped, (2, 0, 1))  # HWC → CHW
        input_tensor = torch.tensor(cropped).unsqueeze(0)

        with torch.inference_mode():
            output = efficient_model(input_tensor)
            predicted = torch.argmax(output, dim=1).item()

        result_label = CLASS_NAMES[predicted]

        return jsonify({"result": result_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ----------------------------
# START
# ----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
