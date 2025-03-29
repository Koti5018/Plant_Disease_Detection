from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import json

# Initialize Flask app
app = Flask(__name__)

# Ensure the 'uploads' folder exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
MODEL_PATH = r"C:\Projects\plant-disease-detection\plant-disease-detection\model\inceptionv3_plant_disease_multi_v2.h5"  # Use raw string
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"⚠️ Model file not found at {MODEL_PATH}! Please check the file path.")

model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels (Updated for 30 classes)
class_labels = [
    "Healthy", "Powdery Mildew", "Rust", "Leaf Spot", "Blight",
    "Bacterial Wilt", "Early Blight", "Late Blight", "Downy Mildew",
    "Anthracnose", "Cercospora Leaf Spot", "Mosaic Virus", "Fusarium Wilt",
    "Verticillium Wilt", "Black Rot", "Alternaria Leaf Spot", "Charcoal Rot",
    "Damping-Off", "Gray Mold", "Septoria Leaf Spot", "Phytophthora Rot",
    "Sooty Mold", "Yellow Leaf Curl Virus", "Stem Canker", "Bacterial Leaf Streak",
    "Root Knot Nematode", "Pythium Root Rot", "Powdery Scab", "Rhizoctonia Root Rot"
]

# Load disease information from JSON file
DISEASE_INFO_PATH = r"C:\Projects\plant-disease-detection\plant-disease-detection\disease_info.json"  # Use raw string
if os.path.exists(DISEASE_INFO_PATH):
    with open(DISEASE_INFO_PATH, "r", encoding="utf-8") as f:
        disease_info = json.load(f)
    print("✅ Disease info loaded successfully! Found", len(disease_info), "diseases.")
else:
    print("⚠️ Disease info file not found!")
    disease_info = {}

# Convert JSON keys to lowercase to ensure matching
disease_info_lower = {k.lower(): v for k, v in disease_info.items()}

# Function to preprocess image
def preprocess_image(img_path):
    """Preprocess the input image for model prediction."""
    img = image.load_img(img_path, target_size=(299, 299))  # Resize for InceptionV3
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array

# Route for Home Page
@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')

# Route for Prediction
@app.route('/predict', methods=['POST'])
def predict():
    """Handles image upload, model prediction, and result rendering."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save uploaded file
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        # Preprocess the image
        img_array = preprocess_image(file_path)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_index]

        # Ensure case-insensitive lookup in JSON
        predicted_class_lower = predicted_class.lower()
        disease_details = disease_info_lower.get(predicted_class_lower, {
            "cure": "No information available.",
            "growth_tips": "No information available."
        })

        # Debugging log
        if disease_details["cure"] == "No information available.":
            print(f"⚠️ No information found for '{predicted_class}'. Please update disease_info.json!")

        return render_template('result.html',
                               prediction=predicted_class,
                               confidence=f"{round(float(np.max(prediction) * 100), 2)}%",
                               cure=disease_details["cure"],
                               growth_tips=disease_details["growth_tips"])

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
