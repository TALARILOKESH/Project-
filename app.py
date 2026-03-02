from flask import Flask, request, render_template_string, send_from_directory
import os
import time
import cv2
import torch
from ultralytics import YOLO
from torchvision import models, transforms
from PIL import Imaget transforms, models
from PIL import Image
from ultralytics import YOLO
from werkzeug.utils import secure_filename
# ================= CONFIG =================
YOLO_MODEL_PATH = "model/best.pt"
EFF_MODEL_PATH = "model/efficientnet_b0_best.pth"

CLASS_MAP = {0: "BAD", 1: "GOOD"}
NUM_CLASSES = 2
CONF_THRESHOLD = 0.75

UPLOAD_DIR = "uploads"
RESULT_DIR = "results"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# =========================================

app = Flask(__name__)

# -------- Load Models ONCE --------
print("Loading YOLO model...")
yolo = YOLO(YOLO_MODEL_PATH)

print("Loading EfficientNet model...")
eff_model = models.efficientnet_b0(weights=None)
eff_model.classifier[1] = torch.nn.Linear(
    eff_model.classifier[1].in_features, NUM_CLASSES
)
eff_model.load_state_dict(torch.load(EFF_MODEL_PATH, map_location="cpu"))
eff_model.eval()

# -------- Preprocessing --------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------- HTML --------
HTML = """
<!doctype html>
<title>Tomato Quality Detection</title>

<h2>🍅 Tomato Quality Detection</h2>

<form method="post" action="/upload" enctype="multipart/form-data">
  <input type="file" name="image" required>
  <br><br>
  <button type="submit">Analyze</button>
</form>

{% if result %}
<hr>
<h3>{{ result }}</h3>
<img src="{{ image_path }}" width="500">
<br><br>
<a href="/">Analyze another image</a>
{% endif %}
"""

# -------- Routes --------
@app.route("/")
def home():
    return render_template_string(HTML)

@app.route("/results/<path:filename>")
def serve_results(filename):
    return send_from_directory(RESULT_DIR, filename)

@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return "<h3>No image selected</h3><a href='/'>Go back</a>"

    file = request.files["image"]
    if file.filename == "":
        return "<h3>No image selected</h3><a href='/'>Go back</a>"

    timestamp = int(time.time() * 1000)
    img_path = os.path.join(UPLOAD_DIR, f"{timestamp}.jpg")
    file.save(img_path)

    frame = cv2.imread(img_path)

    # 🔥 Resize to reduce memory
    frame = cv2.resize(frame, (640, 640))

    results = yolo.predict(
        source=frame,
        imgsz=416,
        conf=0.4,
        verbose=False
    )[0]

    detected = good_count = bad_count = 0

    for box in results.boxes:
        cls_id = int(box.cls[0])

        if yolo.names[cls_id].lower() != "tomato":
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensor = transform(pil_img).unsqueeze(0)

        with torch.no_grad():
            probs = torch.softmax(eff_model(tensor), dim=1)
            pred = torch.argmax(probs, dim=1).item()
            conf = probs[0][pred].item()

        if conf < CONF_THRESHOLD:
            continue

        quality = CLASS_MAP[pred]
        color = (0, 255, 0) if quality == "GOOD" else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{quality} {conf:.2f}",
            (x1, max(y1 - 10, 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

        detected += 1
        if quality == "GOOD":
            good_count += 1
        else:
            bad_count += 1

    if detected == 0:
        return "<h3>No tomatoes detected ❌</h3><a href='/'>Try again</a>"

    final_img = f"final_{timestamp}.jpg"
    cv2.imwrite(f"{RESULT_DIR}/{final_img}", frame)

    return render_template_string(
        HTML,
        result=f"Detected {detected} | GOOD: {good_count} | BAD: {bad_count}",
        image_path=f"/results/{final_img}"
    )