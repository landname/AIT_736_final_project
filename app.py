from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import joblib
import os

app = Flask(__name__)

# ============================================================
# 1. Load Models
# ============================================================

MODEL_DIR = "models"

# Deep learning models
bn_cnn_model = load_model(os.path.join(MODEL_DIR, "bn_cnn_mnist_model.h5"))
aug_cnn_model = load_model(os.path.join(MODEL_DIR, "cnn_with_data_augmentation_final.h5"))

# Traditional machine learning models
logreg_model = joblib.load(os.path.join(MODEL_DIR, "logistic_regression_mnist_model.joblib"))
rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest_mnist_model.joblib"))


# ============================================================
# 2. Image Lists for Result Sections
# ============================================================

bn_cnn_images = [
    {
        "title": "BN CNN Result Table",
        "file": "bn_cnn_result_table.png"
    },
    {
        "title": "BN CNN Classification Report",
        "file": "bn_cnn_classification_report.png"
    },
    {
        "title": "BN CNN Confusion Matrix",
        "file": "bn_cnn_confusion_matrix.png"
    }
]

best_model_images = [
    {
        "title": "Final Best Model Result Table",
        "file": "cnn_with_data_augmentation_final_result_table.png"
    },
    {
        "title": "Final Best Model Classification Report",
        "file": "cnn_with_data_augmentation_final_classification_report.png"
    },
    {
        "title": "Final Best Model Confusion Matrix",
        "file": "cnn_with_data_augmentation_final_confusion_matrix.png"
    },
    {
        "title": "Correct Prediction Examples",
        "file": "cnn_with_data_augmentation_final_correct_predictions.png"
    },
    {
        "title": "Incorrect Prediction Examples",
        "file": "cnn_with_data_augmentation_final_incorrect_predictions.png"
    },
    {
        "title": "Model File Size Table",
        "file": "cnn_with_data_augmentation_final_file_size_table.png"
    }
]

traditional_ml_images = [
    {
        "title": "Logistic Regression Result Table",
        "file": "logistic_regression_result_table.png"
    },
    {
        "title": "Logistic Regression Confusion Matrix",
        "file": "logistic_regression_confusion_matrix.png"
    },
    {
        "title": "Random Forest Result Table",
        "file": "random_forest_result_table.png"
    },
    
    {
        "title": "Random Forest Confusion Matrix",
        "file": "random_forest_confusion_matrix.png"
    }
]


# ============================================================
# 3. Image Preprocessing
# ============================================================

def preprocess_for_cnn(image):
    """
    Preprocess uploaded image for CNN models.
    Output shape: (1, 28, 28, 1)
    """

    image = image.convert("L")
    image = image.resize((28, 28))

    # MNIST has white digits on black background.
    # If user uploads black digit on white background, invert it.
    image = ImageOps.invert(image)

    image_array = np.array(image)
    image_array = image_array.astype("float32") / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)

    return image_array


def preprocess_for_ml(image):
    """
    Preprocess uploaded image for Logistic Regression and Random Forest.
    Output shape: (1, 784)
    """

    image = image.convert("L")
    image = image.resize((28, 28))
    image = ImageOps.invert(image)

    image_array = np.array(image)
    image_array = image_array.astype("float32") / 255.0
    image_array = image_array.reshape(1, 784)

    return image_array


# ============================================================
# 4. Routes
# ============================================================

@app.route("/")
def home():
    return render_template(
        "index.html",
        bn_cnn_images=bn_cnn_images,
        best_model_images=best_model_images,
        traditional_ml_images=traditional_ml_images,
        predictions=None
    )


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template(
            "index.html",
            prediction_error="No file uploaded",
            bn_cnn_images=bn_cnn_images,
            best_model_images=best_model_images,
            traditional_ml_images=traditional_ml_images,
            predictions=None
        )

    file = request.files["file"]

    if file.filename == "":
        return render_template(
            "index.html",
            prediction_error="No file selected",
            bn_cnn_images=bn_cnn_images,
            best_model_images=best_model_images,
            traditional_ml_images=traditional_ml_images,
            predictions=None
        )

    try:
        image = Image.open(file)

        cnn_input = preprocess_for_cnn(image)
        ml_input = preprocess_for_ml(image)

        predictions = []

        # -----------------------------
        # Best Model: CNN with Data Augmentation
        # -----------------------------
        aug_probs = aug_cnn_model.predict(cnn_input, verbose=0)
        aug_pred = int(np.argmax(aug_probs, axis=1)[0])
        aug_conf = float(np.max(aug_probs) * 100)

        predictions.append({
            "model": "CNN with Data Augmentation",
            "type": "Best Deep Learning Model",
            "prediction": aug_pred,
            "confidence": round(aug_conf, 2)
        })

        # -----------------------------
        # BN CNN
        # -----------------------------
        bn_probs = bn_cnn_model.predict(cnn_input, verbose=0)
        bn_pred = int(np.argmax(bn_probs, axis=1)[0])
        bn_conf = float(np.max(bn_probs) * 100)

        predictions.append({
            "model": "BN CNN",
            "type": "Deep Learning Model",
            "prediction": bn_pred,
            "confidence": round(bn_conf, 2)
        })

        # -----------------------------
        # Logistic Regression
        # -----------------------------
        logreg_pred = int(logreg_model.predict(ml_input)[0])

        if hasattr(logreg_model, "predict_proba"):
            logreg_probs = logreg_model.predict_proba(ml_input)
            logreg_conf = float(np.max(logreg_probs) * 100)
        else:
            logreg_conf = None

        predictions.append({
            "model": "Logistic Regression",
            "type": "Traditional Machine Learning",
            "prediction": logreg_pred,
            "confidence": round(logreg_conf, 2) if logreg_conf is not None else "N/A"
        })

        # -----------------------------
        # Random Forest
        # -----------------------------
        rf_pred = int(rf_model.predict(ml_input)[0])

        if hasattr(rf_model, "predict_proba"):
            rf_probs = rf_model.predict_proba(ml_input)
            rf_conf = float(np.max(rf_probs) * 100)
        else:
            rf_conf = None

        predictions.append({
            "model": "Random Forest",
            "type": "Traditional Machine Learning",
            "prediction": rf_pred,
            "confidence": round(rf_conf, 2) if rf_conf is not None else "N/A"
        })

        return render_template(
            "index.html",
            predictions=predictions,
            bn_cnn_images=bn_cnn_images,
            best_model_images=best_model_images,
            traditional_ml_images=traditional_ml_images,
            prediction_error=None
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_error=f"Error: {str(e)}",
            bn_cnn_images=bn_cnn_images,
            best_model_images=best_model_images,
            traditional_ml_images=traditional_ml_images,
            predictions=None
        )


if __name__ == "__main__":
    app.run(debug=True)