from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    color_class = None
    image_path = None

    if request.method == 'POST':
        file = request.files['image']

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            image_path = filepath

            # 🔹 Replace this with your model prediction
            prediction = "Good"   # or "Bad"

            if prediction == "Good":
                color_class = "good"
            else:
                color_class = "bad"

    return render_template("index.html",
                           prediction=prediction,
                           color_class=color_class,
                           image_path=image_path)

if __name__ == '__main__':
    app.run()