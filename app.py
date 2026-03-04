from flask import Flask, request, send_file
from flask_cors import CORS
from ultralytics import YOLO
import torch
import cv2
import numpy as np
import io
import os

# ----------------------------
# CPU CONTROL (CRITICAL)
# ----------------------------
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

torch.set_grad_enabled(False)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

cv2.setNumThreads(0)
cv2.setUseOptimized(True)

# ----------------------------
# APP
# ----------------------------
app = Flask(__name__)
CORS(app)

IMAGE_SIZE_YOLO = 192
IMAGE_SIZE_EFF = 224

CLASS_MAP = {0: "BAD", 1: "GOOD"}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_PATH = os.path.join(BASE_DIR, "model", "best.pt")
EFF_PATH = os.path.join(BASE_DIR, "model", "efficientnet_scripted.pt")

# ----------------------------
# LOAD MODELS ONCE
# ----------------------------
print("Loading YOLO...")
yolo = YOLO(YOLO_PATH)
yolo.model.eval()

print("Loading EfficientNet...")
efficient_model = torch.jit.load(EFF_PATH, map_location="cpu")
efficient_model.eval()

print("Models loaded")

# ----------------------------
# ROUTE
# ----------------------------
@app.route("/")
def home():
    return "Tomato Sorting Backend Running"


@app.route("/detect", methods=["POST"])
def detect():

    if "image" not in request.files:
        return "No image", 400

    # ----------------------------
    # FAST IMAGE LOAD
    # ----------------------------
    file_bytes = request.files["image"].read()
    np_img = np.frombuffer(file_bytes, np.uint8)

    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # ----------------------------
    # SMART RESIZE
    # ----------------------------
    h, w = image.shape[:2]

    MAX_DIM = 512
    scale = MAX_DIM / max(h, w)

    if scale < 1:
        image = cv2.resize(image, (int(w*scale), int(h*scale)))

    original = image.copy()

    # ----------------------------
    # YOLO DETECTION
    # ----------------------------
    with torch.inference_mode():

        results = yolo.predict(
            image,
            imgsz=IMAGE_SIZE_YOLO,
            conf=0.25,
            device="cpu",
            verbose=False
        )

    if len(results[0].boxes) == 0:
        return "No Tomato Detected"

    boxes = results[0].boxes.xyxy.numpy()

    # ----------------------------
    # PREPARE CROPS
    # ----------------------------
    crops = []
    coords = []

    for b in boxes:

        x1,y1,x2,y2 = b.astype(int)

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crop = cv2.resize(crop,(IMAGE_SIZE_EFF,IMAGE_SIZE_EFF))

        crop = crop.astype(np.float32)/255.0
        crop = np.transpose(crop,(2,0,1))

        crops.append(crop)
        coords.append((x1,y1,x2,y2))

    if len(crops)==0:
        return "No valid crop"

    # ----------------------------
    # BATCH CLASSIFICATION
    # ----------------------------
    batch = torch.from_numpy(np.array(crops))

    with torch.inference_mode():
        out = efficient_model(batch)

    probs = torch.softmax(out,dim=1)
    preds = torch.argmax(probs,dim=1)

    # ----------------------------
    # DRAW RESULTS
    # ----------------------------
    for i,(x1,y1,x2,y2) in enumerate(coords):

        p = preds[i].item()
        prob = probs[i][p].item()

        label = f"Tomato: {CLASS_MAP[p]} ({prob:.2f})"

        color = (0,0,255) if p==1 else (0,255,0)

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

    # ----------------------------
    # FAST RETURN
    # ----------------------------
    _,buffer = cv2.imencode(
        ".jpg",
        cv2.cvtColor(original,cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY),80]
    )

    return send_file(io.BytesIO(buffer),mimetype="image/jpeg")


if __name__=="__main__":
    app.run(host="0.0.0.0",port=10000)