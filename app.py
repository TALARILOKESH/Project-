import os
from flask import Flask, request, jsonify
from ultralytics import YOLO
import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🔥 USE LIGHTWEIGHT YOLO (not your heavy best.pt)
YOLO_PATH = "yolov8n.pt"

# ✅ KEEP YOUR EfficientNet PATH EXACTLY SAME
EFF_PATH = os.path.join(BASE_DIR, "model", "efficientnet_b0_best.pth")

# 🔥 Disable gradients (huge RAM save)
torch.set_grad_enabled(False)
device = torch.device("cpu")

# ==========================
# 1️⃣ LOAD YOLO (Nano Version)
# ==========================
yolo_model = YOLO(YOLO_PATH)
yolo_model.fuse = False   # prevent memory spike

# ==========================
# 2️⃣ LOAD YOUR EfficientNet (.pth)
# ==========================
classifier_model = models.efficientnet_b0(weights=None)

classifier_model.classifier[1] = nn.Linear(
    classifier_model.classifier[1].in_features,
    2
)

classifier_model.load_state_dict(
    torch.load(EFF_PATH, map_location=device)
)

classifier_model.to(device)
classifier_model.eval()

# ==========================
# 3️⃣ IMAGE TRANSFORM
# ==========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ==========================
# 4️⃣ ROUTE
# ==========================
@app.route("/detect", methods=["POST"])
def detect():
    try:
        file = request.files["image"]

        file_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # 🔥 Reduce YOLO input size
        img_resized = cv2.resize(img, (512, 512))

        results = yolo_model(img_resized, imgsz=512, verbose=False)

        output = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                crop = img[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                crop_pil = Image.fromarray(
                    cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                )

                input_tensor = transform(crop_pil).unsqueeze(0).to(device)

                with torch.no_grad():
                    prediction = classifier_model(input_tensor)
                    predicted_class = torch.argmax(prediction, dim=1).item()

                label = "Good" if predicted_class == 0 else "Bad"

                output.append({"label": label})

        return jsonify({"results": output})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)