from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import io

# Load the trained model
model = tf.keras.models.load_model('sign_language_model.h5')

# Load the class names (sign labels)
classes = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', 
    '4', '5', '6', '7', '8', '9'
]

# Initialize Flask app
app = Flask(__name__)

# Function to preprocess and set background to black
def preprocess_image(image_array):
    # Convert image to grayscale
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    
    # Apply a binary threshold to isolate the hand
    _, thresholded = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Create a mask for the hand (white area) and black out the background
    background_black = cv2.bitwise_and(image_array, image_array, mask=thresholded)
    
    return background_black

# Flask route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle video frame (ROI) and make predictions continuously (live feed)
@app.route('/predict_live', methods=['POST'])
def predict_live():
    # Get the image from the POST request
    file = request.files['frame'].read()

    # Convert the file bytes to an image
    img = Image.open(io.BytesIO(file))
    img = img.resize((64, 64))  # Resize the ROI image to match the model input size

    # Convert the image to a numpy array
    img_array = np.array(img)

    # Preprocess the image: Set background to black
    img_array = preprocess_image(img_array)

    # Preprocess for prediction
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize

    # Make predictions
    predictions = model.predict(img_array)
    pred_class = np.argmax(predictions)  # Get the predicted class index
    sign_label = classes[pred_class]  # Map the predicted class to the actual label

    # Return the prediction as a JSON response
    return jsonify(prediction=sign_label)

# Route to handle a single image capture and make prediction (button click)
@app.route('/predict_image', methods=['POST'])
def predict_image():
    # Get the image from the POST request
    file = request.files['image'].read()

    # Convert the file bytes to an image
    img = Image.open(io.BytesIO(file))
    img = img.resize((64, 64))  # Resize the image to match the model input size

    # Convert the image to a numpy array
    img_array = np.array(img)

    # Preprocess the image: Set background to black
    img_array = preprocess_image(img_array)

    # Preprocess for prediction
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize

    # Make predictions
    predictions = model.predict(img_array)
    pred_class = np.argmax(predictions)  # Get the predicted class index
    sign_label = classes[pred_class]  # Map the predicted class to the actual label

    # Return the prediction as a JSON response
    return jsonify(prediction=sign_label)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
