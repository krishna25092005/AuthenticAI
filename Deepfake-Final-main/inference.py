import gdown
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Google Drive file ID (Replace with your actual file ID)
model_path = "final-model-1.h5"
download_url = f"https://drive.google.com/file/d/1NfoLyaOQwKov3Y2mEMP1DfMrKLhLR722/view?usp=sharing"

# Ensure the model is downloaded and valid
if not os.path.exists(model_path) or os.path.getsize(model_path) < 10_000_000:  # Check if the file is at least 10MB
    print("Downloading model from Google Drive...")
    gdown.download(download_url, model_path, quiet=False)

    # Check if the download was successful
    if os.path.getsize(model_path) < 10_000_000:  
        raise Exception("Model download failed or is incomplete. Check the Google Drive link.")

# Load the model
model = load_model(model_path)

def preprocess_image(image):
    img = image.resize((224, 224))  # Resize image for model input
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict(image):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    return prediction
