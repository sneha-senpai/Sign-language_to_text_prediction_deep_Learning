from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import cv2
import base64
import sys

app = Flask(__name__)
CORS(app)

# Load pre-trained models
model = load_model('sign_language_model.h5')

with open('label_encoder.pkl', 'rb') as f:
    additional_model = pickle.load(f)

# Function to preprocess the input image (resize, normalize, etc.)
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))  # Resize to match model input size
    image = image / 255.0  # Normalize the pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'image' not in data:
        return jsonify({"error": "No image data provided"}), 400

    # Decode base64 image
    image_data = base64.b64decode(data['image'].split(',')[1])
    npimg = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Make prediction using the .h5 model
    prediction = model.predict(processed_image)

    # Optionally, use the additional .pkl model for post-processing (e.g., decoding prediction)
    result = additional_model.transform(prediction)  # Example usage of the .pkl model

    # Return the result as JSON
    return jsonify({"prediction": result.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
