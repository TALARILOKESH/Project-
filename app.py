from flask import Flask, request, send_file
from flask_cors import CORS
from ultralytics import YOLO
import torch
import cv2
import numpy as np
from PIL import Image
import io
import os

# ----------------------------
# SPEED OPTIMIZATION
# ----------------------------
torch.set_grad_enabled(False)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# ----------------------------
# APP INIT
# ----------------------------
app = Flask(__name__)
CORS(app)

IMAGE_SIZE_YOLO = 192   # reduced for speed
IMAGE_SIZE_EFF = 224
CLASS_NAMES = ["BAD", "GOOD"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_PATH = os.path.join(BASE_DIR, "model", "best.pt")
EFF_PATH = os.path.join(BASE_DIR, "model", "efficientnet_scripted.pt")

# ----------------------------
# LOAD MODELS
# ----------------------------
print("Loading YOLO...")
yolo_model = YOLO(YOLO_PATH)
yolo_model.to("cpu")
yolo_model.fuse()
yolo_model.model.eval()

print("Loading EfficientNet...")
efficient_model = torch.jit.load(EFF_PATH, map_location="cpu")
efficient_model.eval()

print("Models Ready ✅")

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

    # 🔥 Resize FULL image before YOLO (big speed gain)
    image_np = cv2.resize(image_np, (512, 512))

    original_image = image_np.copy()

    # YOLO detection
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

        # Faster preprocessing
        cropped = cv2.resize(cropped, (IMAGE_SIZE_EFF, IMAGE_SIZE_EFF))
        cropped = cropped.astype(np.float32) / 255.0
        cropped = np.transpose(cropped, (2, 0, 1))
        input_tensor = torch.from_numpy(cropped).unsqueeze(0)

        # Faster inference (no softmax first)
        output = efficient_model(input_tensor)
        confidence, predicted = torch.max(output, dim=1)

        predicted_class = predicted.item()

        # Optional: compute real probability only if needed
        prob = torch.softmax(output, dim=1)[0][predicted_class].item()

        label = f"Tomato: {CLASS_NAMES[predicted_class]} ({prob:.2f})"

        color = (0, 0, 255) if predicted_class == 0 else (0, 255, 0)

        cv2.rectangle(original_image, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            original_image,
            label,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    result_image = Image.fromarray(original_image)
    img_io = io.BytesIO()
    result_image.save(img_io, format="JPEG", quality=85)
    img_io.seek(0)

    return send_file(img_io, mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)