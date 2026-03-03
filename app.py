from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)

# -----------------------------
# Load YOLO Model
# -----------------------------
yolo_model = YOLO("yolo_best.pt")

# -----------------------------
# Load EfficientNet Model
# -----------------------------
class_names = ["Bad Tomato", "Good Tomato"]  # Change if needed

device = torch.device("cpu")

efficient_model = torch.load("efficientnet_b0_best.pth", map_location=device)
efficient_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -----------------------------
# API Route
# -----------------------------
@app.route("/detect", methods=["POST"])
def detect():

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"})

    file = request.files["image"].read()
    image = Image.open(io.BytesIO(file)).convert("RGB")
    image_np = np.array(image)

    # -------- YOLO Detection --------
    results = yolo_model(image_np)

    if len(results[0].boxes) == 0:
        return jsonify({"result": "No Tomato Detected"})

    # Get first detected box
    box = results[0].boxes.xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = map(int, box)

    cropped = image_np[y1:y2, x1:x2]

    # -------- EfficientNet Classification --------
    cropped_pil = Image.fromarray(cropped)
    input_tensor = transform(cropped_pil).unsqueeze(0)

    with torch.no_grad():
        output = efficient_model(input_tensor)
        predicted = torch.argmax(output, 1).item()

    result_label = class_names[predicted]

    return jsonify({
        "result": result_label
    })


if __name__ == "__main__":
    app.run()