import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the model
model = load_model('/mount/src/authenticai/Deepfake-Final-main/final-model-1.h5')  # Adjust path if needed

def preprocess_image(image):
    # Convert RGBA (4 channels) or Grayscale (1 channel) to RGB (3 channels)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    img = image.resize((224, 224))  # Resize image for model input
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict(image):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    return prediction
