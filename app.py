from flask import Flask, render_template, request
from ultralytics import YOLO
import os
import cv2
import torch
import numpy as np
from torchvision import transforms, models
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["PROCESSED_FOLDER"] = PROCESSED_FOLDER

# -------------------------------
# 🔥 LOAD YOLO MODEL
# -------------------------------
yolo_model = torch.hub.load(
    'ultralytics/yolov5',
    'custom',
    path='model/best.pt',
    force_reload=False
)

# -------------------------------
# 🔥 LOAD EFFICIENTNET MODEL
# -------------------------------
classifier = models.efficientnet_b0(pretrained=False)
classifier.classifier[1] = torch.nn.Linear(classifier.classifier[1].in_features, 2)
classifier.load_state_dict(torch.load('model/efficientnet_b0_best.pth', map_location='cpu'))
classifier.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------------
# ROUTE
# -------------------------------
@app.route('/', methods=['GET', 'POST'])
def index():
    image_path = None
    processed_path = None
    good_count = 0
    bad_count = 0

    if request.method == 'POST':
        file = request.files['image']

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            image_path = filepath
            img = cv2.imread(filepath)

            # 🔥 YOLO Detection
            results = yolo_model(img)
            boxes = results.xyxy[0]

            for box in boxes:
                x1, y1, x2, y2, conf, cls = box.tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                crop = img[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                crop_tensor = transform(crop_pil).unsqueeze(0)

                with torch.no_grad():
                    output = classifier(crop_tensor)
                    _, pred = torch.max(output, 1)

                # 0 = Good , 1 = Bad (adjust if needed)
                if pred.item() == 0:
                    label = "Good"
                    color = (0, 255, 0)
                    good_count += 1
                else:
                    label = "Bad"
                    color = (0, 0, 255)
                    bad_count += 1

                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, color, 2)

            # Save processed image
            processed_filename = "processed_" + filename
            processed_filepath = os.path.join(app.config["PROCESSED_FOLDER"], processed_filename)
            cv2.imwrite(processed_filepath, img)

            processed_path = processed_filepath

    return render_template("index.html",
                           image_path=image_path,
                           processed_path=processed_path,
                           good_count=good_count,
                           bad_count=bad_count)

if __name__ == "__main__":
    app.run()