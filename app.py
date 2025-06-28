from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model("rice_type_classifier.h5")
classes = ['Basmati', 'Jasmine', 'Arborio', 'Brown', 'Red']

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        img_file = request.files["file"]
        img_path = os.path.join("static", img_file.filename)
        img_file.save(img_path)

        img = image.load_img(img_path, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        prediction = model.predict(img)
        label = classes[np.argmax(prediction)]

        return render_template("index.html", prediction=label, image_path=img_path)

    return render_template("index.html", prediction="", image_path="")

if __name__ == "__main__":
    app.run(debug=True)
