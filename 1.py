import sys
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import os

# Ensure UTF-8 encoding to avoid Unicode errors
sys.stdout.reconfigure(encoding="utf-8")

# Load the pre-trained InceptionV3 model (without top layers)
base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(299, 299, 3))

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Reduces the dimensionality
x = Dense(2048, activation="relu")(x)  # Increased neurons for better feature extraction
x = Dropout(0.4)(x)  # Increased dropout to prevent overfitting
x = Dense(1024, activation="relu")(x)
x = Dropout(0.4)(x)

# Output layer (change number of classes based on your dataset)
num_classes = 30  # Example: 29 plant diseases + 1 healthy plant
output = Dense(num_classes, activation="softmax")(x)

# Create and compile the model
model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=0.00005), loss="categorical_crossentropy", metrics=["accuracy"])

# Define model save path
MODEL_DIR = r"C:/Users/Lenovo/OneDrive/Desktop/Koti Project/Project2/plant-disease-detection/plant-disease-detection/model"
MODEL_PATH = os.path.join(MODEL_DIR, "inceptionv3_plant_disease_multi_v2.h5")

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Save the improved model
model.save(MODEL_PATH)

print(f"✅ Model successfully saved at: {MODEL_PATH}")
