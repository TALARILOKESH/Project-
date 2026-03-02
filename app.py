from flask import Flask, render_template, request
import random

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    color_class = None

    if request.method == 'POST':
        file = request.files['image']

        # 🔹 Replace this with your model prediction
        prediction = random.choice(["Good", "Bad"])

        if prediction == "Good":
            color_class = "good"
        else:
            color_class = "bad"

    return render_template("index.html",
                           prediction=prediction,
                           color_class=color_class)

if __name__ == '__main__':
    app.run()