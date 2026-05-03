from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import os

app = Flask(__name__)

# Load trained MNIST CNN model
model = load_model("mnist_cnn_model.h5")


def preprocess_image(image):
    """
    Preprocess uploaded image to match MNIST CNN input:
    shape: (1, 28, 28, 1)
    pixel range: 0 to 1
    """

    # Convert to grayscale
    image = image.convert("L")

    # Resize to 28x28
    image = image.resize((28, 28))

    # Invert image if background is white
    # MNIST digits are white digits on black background
    image = ImageOps.invert(image)

    # Convert to numpy array
    image_array = np.array(image)

    # Normalize pixel values
    image_array = image_array.astype("float32") / 255.0

    # Reshape for CNN
    image_array = image_array.reshape(1, 28, 28, 1)

    return image_array


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", prediction="No file uploaded")

    file = request.files["file"]

    if file.filename == "":
        return render_template("index.html", prediction="No file selected")

    try:
        image = Image.open(file)

        processed_image = preprocess_image(image)

        prediction_probs = model.predict(processed_image)
        predicted_digit = np.argmax(prediction_probs, axis=1)[0]
        confidence = np.max(prediction_probs) * 100

        return render_template(
            "index.html",
            prediction=predicted_digit,
            confidence=round(confidence, 2)
        )

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)