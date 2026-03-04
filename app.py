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

# ----------------------------
# APP INIT
# ----------------------------
app = Flask(__name__)
CORS(app)

# ----------------------------
# CONFIG
# ----------------------------
IMAGE_SIZE_YOLO = 256
IMAGE_SIZE_EFF = 224

# IMPORTANT:
# 0 = Bad
# 1 = Good
CLASS_NAMES = ["BAD", "GOOD"]

# ----------------------------
# PATHS
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_PATH = os.path.join(BASE_DIR, "model", "best.pt")
EFF_PATH = os.path.join(BASE_DIR, "model", "efficientnet_scripted.pt")

# ----------------------------
# LOAD MODELS
# ----------------------------
print("Loading YOLO model...")
yolo_model = YOLO(YOLO_PATH)
yolo_model.to("cpu")
yolo_model.fuse()
yolo_model.model.eval()

print("Loading EfficientNet...")
efficient_model = torch.jit.load(EFF_PATH, map_location="cpu")
efficient_model.eval()

print("Models Loaded Successfully ✅")

# ----------------------------
# ROUTES
# ----------------------------
@app.route("/")
def home():
    return "Tomato Sorting Backend Running ✅"


@app.route("/detect", methods=["POST"])
def detect():

    if "image" not in request.files:
        return "No image uploaded", 400

    file_bytes = request.files["image"].read()
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    image_np = np.array(image)

    original_image = image_np.copy()

    # YOLO Detection
    with torch.inference_mode():
        results = yolo_model(
            image_np,
            imgsz=IMAGE_SIZE_YOLO,
            verbose=False,
            device="cpu"
        )

    if len(results[0].boxes) == 0:
        return "No Tomato Detected", 200

    boxes = results[0].boxes.xyxy.cpu().numpy()

    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)

        cropped = image_np[y1:y2, x1:x2]

        if cropped.size == 0:
            continue

        # Classification
        cropped_resized = cv2.resize(cropped, (IMAGE_SIZE_EFF, IMAGE_SIZE_EFF))
        cropped_resized = cropped_resized.astype("float32") / 255.0
        cropped_resized = np.transpose(cropped_resized, (2, 0, 1))
        input_tensor = torch.from_numpy(cropped_resized).unsqueeze(0)

        with torch.inference_mode():
            output = efficient_model(input_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probs, dim=1)

        predicted_class = predicted.item()
        confidence_score = confidence.item()

        label = f"Tomato: {CLASS_NAMES[predicted_class]} ({confidence_score:.2f})"

        # Color: Red for BAD, Green for GOOD
        if predicted_class == 0:
            color = (0, 0, 255)  # Red
        else:
            color = (0, 255, 0)  # Green

        # Draw bounding box
        cv2.rectangle(original_image, (x1, y1), (x2, y2), color, 2)

        # Draw text
        cv2.putText(
            original_image,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    # Convert to image for response
    result_image = Image.fromarray(original_image)
    img_io = io.BytesIO()
    result_image.save(img_io, format="JPEG")
    img_io.seek(0)

    gc.collect()

    return send_file(img_io, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)