import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image

# Initialize Flask app
app = Flask(__name__)

# Load trained model
model = tf.keras.models.load_model("model/crop_model.h5")

# Class names (must match dataset folder names exactly)
class_names = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy"
]

# Upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Image preprocessing
def prepare_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None
    probabilities = None
    status = None
    status_color = None
    recommendation = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(image_path)

            img_array = prepare_image(image_path)
            preds = model.predict(img_array)

            predicted_index = np.argmax(preds)
            predicted_class = class_names[predicted_index]
            confidence = round(float(np.max(preds) * 100), 2)

            # Probability dictionary for chart
            probabilities = {
                class_names[i]: round(float(preds[0][i] * 100), 2)
                for i in range(len(class_names))
            }

            prediction = predicted_class

            # Status Logic
            if "healthy" in predicted_class.lower():
                status = "Healthy"
                status_color = "#00ffaa"
                recommendation = "No disease detected. The plant is healthy and safe."
            else:
                status = "Diseased"
                status_color = "#ff4d4d"
                recommendation = "Disease detected. Immediate attention and treatment recommended."

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path,
        probabilities=probabilities,
        status=status,
        status_color=status_color,
        recommendation=recommendation
    )


if __name__ == "__main__":
    app.run(debug=True)