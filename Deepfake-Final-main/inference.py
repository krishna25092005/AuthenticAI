# inference.py
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the model
model = load_model('final-model-1.h5')

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
