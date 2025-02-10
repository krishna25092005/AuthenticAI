# inference.py
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import gdown
import os

# Google Drive model URL (Replace YOUR_FILE_ID)
url = "https://drive.google.com/file/d/1TDXsknpGlWghk1mxiN4bJQ9t3EBvn7eG/view?usp=sharing"
model_path = "final-model-1.h5"

# Download the model if it doesn't exist
if not os.path.exists(model_path):
    print("Downloading model from Google Drive...")
    gdown.download(url, model_path, quiet=False)

# Load the model
model = load_model(model_path)

def preprocess_image(image):
    image = image.convert("RGB")  # Convert RGBA/Grayscale to RGB
    img = image.resize((224, 224))  # Resize according to model's input shape
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array


def predict(image):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    return prediction
