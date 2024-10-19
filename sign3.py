# import cv2
# import numpy as np
# import tensorflow as tf
# import joblib
# from tkinter import Tk, Label, Button, Frame, filedialog
# from PIL import Image, ImageTk

# # Load the trained model
# model = tf.keras.models.load_model('sign_language_model.h5')

# # Load Label Encoder for decoding predictions
# label_encoder = joblib.load('label_encoder.pkl')

# # Function to preprocess the frame for prediction
# def preprocess_frame(frame):
#     # Resize frame to 32x32 (small window size for hand)
#     frame_resized = cv2.resize(frame, (32, 32))
#     # Normalize pixel values to 0-1
#     frame_resized = frame_resized / 255.0
#     # Expand dimensions to match model input shape (1, 32, 32, 3)
#     frame_resized = np.expand_dims(frame_resized, axis=0)
#     return frame_resized

# # Function to predict the sign from a captured frame
# def predict_sign(frame):
#     # Preprocess the frame
#     preprocessed_frame = preprocess_frame(frame)

#     # Predict the class
#     prediction = model.predict(preprocessed_frame)
#     predicted_class = np.argmax(prediction)

#     # Decode the predicted class
#     predicted_label = label_encoder.inverse_transform([predicted_class])[0]

#     return predicted_label

# # Function to apply a mask that filters the background, keeping only the hand visible
# def apply_black_background(frame):
#     # Convert frame to HSV (Hue, Saturation, Value) color space
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     # Define skin color range in HSV
#     lower_skin = np.array([0, 30, 60], dtype=np.uint8)
#     upper_skin = np.array([20, 150, 255], dtype=np.uint8)

#     # Create a binary mask where skin color is white and the rest is black
#     mask = cv2.inRange(hsv, lower_skin, upper_skin)

#     # Apply the mask to the frame
#     hand_region = cv2.bitwise_and(frame, frame, mask=mask)

#     # Create a black background where the mask is not applied
#     background = np.zeros_like(frame)
#     final_frame = np.where(hand_region == 0, background, hand_region)

#     return final_frame

# # Function to capture a snapshot, apply black background filter, and make prediction
# def capture_snapshot():
#     global cap, canvas_label, word_label

#     # Capture a frame from the webcam
#     ret, frame = cap.read()
#     if ret:

#         # frame = cv2.flip(frame, 1)
#         # Define Region of Interest (ROI)
#         roi = frame[50:250, 50:250]  # 200x200 window for hand
#         roi = cv2.flip(roi, 1)
#         # Apply black background filter
#         roi_filtered = apply_black_background(roi)

#         # Predict the sign from the ROI with black background
#         predicted_label = predict_sign(roi_filtered)

#         # Update the label with the predicted sign
#         word_label.config(text=f"Predicted Sign: {predicted_label}")

#         # Convert captured frame (ROI with black background) to display in Tkinter
#         roi_rgb = cv2.cvtColor(roi_filtered, cv2.COLOR_BGR2RGB)
#         img = Image.fromarray(roi_rgb)
#         imgtk = ImageTk.PhotoImage(image=img)

#         # Display the snapshot on the canvas
#         canvas_label.imgtk = imgtk
#         canvas_label.configure(image=imgtk)

# # Function to upload an image, apply black background filter, and make prediction
# def upload_image():
#     global canvas_label, word_label

#     # Open file dialog to select an image file
#     file_path = filedialog.askopenfilename()

#     if file_path:
#         # Load the image from the selected file
#         image = cv2.imread(file_path)

#         # Resize and extract the Region of Interest (ROI)
#         roi = image[50:250, 50:250]  # Assuming similar 200x200 window for hand

#         # Apply black background filter
#         roi_filtered = apply_black_background(roi)

#         # Predict the sign from the ROI with black background
#         predicted_label = predict_sign(roi_filtered)

#         # Update the label with the predicted sign
#         word_label.config(text=f"Predicted Sign: {predicted_label}")

#         # Convert the image (ROI with black background) to display in Tkinter
#         roi_rgb = cv2.cvtColor(roi_filtered, cv2.COLOR_BGR2RGB)
#         img = Image.fromarray(roi_rgb)
#         imgtk = ImageTk.PhotoImage(image=img)

#         # Display the uploaded image on the canvas
#         canvas_label.imgtk = imgtk
#         canvas_label.configure(image=imgtk)

# # Function to update the real-time webcam feed
# def update_webcam_feed():
#     global cap, live_feed_label

#     # Capture the frame from webcam
#     ret, frame = cap.read()
#     if ret:
#         # Draw a rectangle to show the ROI
#         cv2.rectangle(frame, (50, 50), (250, 250), (255, 0, 0), 2)
#         cv2.putText(frame, "Place hand here", (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#         # Convert frame to RGB for displaying in Tkinter
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         img = Image.fromarray(frame_rgb)
#         imgtk = ImageTk.PhotoImage(image=img)

#         # Update the live webcam feed in the label
#         live_feed_label.imgtk = imgtk
#         live_feed_label.configure(image=imgtk)

#     # Repeat the function after 10 milliseconds
#     live_feed_label.after(10, update_webcam_feed)

# # Function to automatically capture and predict every 5 seconds
# def auto_predict():
#     capture_snapshot()
#     root.after(2000, auto_predict)  # Schedule the function to be called after 5 seconds

# # Initialize the webcam
# cap = cv2.VideoCapture(0)

# # Create the Tkinter window
# root = Tk()
# root.title("Sign Language Detection - Webcam Snapshot")

# # Create a frame for the live feed and the prediction side-by-side
# main_frame = Frame(root)
# main_frame.pack()

# # Create a label to display the real-time webcam feed
# live_feed_label = Label(main_frame)
# live_feed_label.grid(row=0, column=0)

# # Create a frame for the prediction label and the snapshot
# prediction_frame = Frame(main_frame)
# prediction_frame.grid(row=0, column=1, padx=10)

# # Create a label to display the predicted word
# word_label = Label(prediction_frame, text="Predicted Sign: ", font=("Helvetica", 16))
# word_label.pack(pady=10)

# # Label to display the snapshot with the black background, placed below the predicted sign
# canvas_label = Label(prediction_frame)
# canvas_label.pack()

# # Create a button to capture a snapshot and predict
# capture_button = Button(root, text="Capture and Predict", command=capture_snapshot)
# capture_button.pack(pady=10)

# # Create a button to upload an image and predict
# upload_button = Button(root, text="Upload Image", command=upload_image)
# upload_button.pack(pady=10)

# # Start updating the real-time webcam feed
# update_webcam_feed()

# # Start automatic prediction every 5 seconds
# auto_predict()

# # Start the Tkinter main loop
# root.mainloop()

# # Release the webcam after closing the GUI
# cap.release()


import cv2
import numpy as np
import tensorflow as tf
import joblib
from tkinter import Tk, Label, Button, Frame, filedialog
from PIL import Image, ImageTk

# Load the trained model
model = tf.keras.models.load_model('sign_language_model.h5')

# Load Label Encoder for decoding predictions
label_encoder = joblib.load('label_encoder.pkl')

# Function to preprocess the frame for prediction (Updated to 64x64)
def preprocess_frame(frame):
    # Resize frame to 64x64 (to match the model's input size)
    frame_resized = cv2.resize(frame, (64, 64))
    # Normalize pixel values to 0-1
    frame_resized = frame_resized / 255.0
    # Expand dimensions to match model input shape (1, 64, 64, 3)
    frame_resized = np.expand_dims(frame_resized, axis=0)
    return frame_resized

# Function to predict the sign from a captured frame
def predict_sign(frame):
    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)

    # Predict the class
    prediction = model.predict(preprocessed_frame)
    predicted_class = np.argmax(prediction)

    # Decode the predicted class
    predicted_label = label_encoder.inverse_transform([predicted_class])[0]

    return predicted_label

# Function to apply a mask that filters the background, keeping only the hand visible
def apply_black_background(frame):
    # Convert frame to HSV (Hue, Saturation, Value) color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define skin color range in HSV
    lower_skin = np.array([0, 30, 60], dtype=np.uint8)
    upper_skin = np.array([20, 150, 255], dtype=np.uint8)

    # Create a binary mask where skin color is white and the rest is black
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Apply the mask to the frame
    hand_region = cv2.bitwise_and(frame, frame, mask=mask)

    # Create a black background where the mask is not applied
    background = np.zeros_like(frame)
    final_frame = np.where(hand_region == 0, background, hand_region)

    return final_frame

# Function to capture a snapshot, apply black background filter, and make prediction
def capture_snapshot():
    global cap, canvas_label, word_label

    # Capture a frame from the webcam
    ret, frame = cap.read()
    if ret:
        # Define Region of Interest (ROI)
        roi = frame[50:250, 50:250]  # 200x200 window for hand
        roi = cv2.flip(roi, 1)

        # Apply black background filter
        roi_filtered = apply_black_background(roi)

        # Predict the sign from the ROI with black background
        predicted_label = predict_sign(roi_filtered)

        # Update the label with the predicted sign
        word_label.config(text=f"Predicted Sign: {predicted_label}")

        # Convert captured frame (ROI with black background) to display in Tkinter
        roi_rgb = cv2.cvtColor(roi_filtered, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(roi_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        # Display the snapshot on the canvas
        canvas_label.imgtk = imgtk
        canvas_label.configure(image=imgtk)

# Function to upload an image, apply black background filter, and make prediction
def upload_image():
    global canvas_label, word_label

    # Open file dialog to select an image file
    file_path = filedialog.askopenfilename()

    if file_path:
        # Load the image from the selected file
        image = cv2.imread(file_path)

        # Resize and extract the Region of Interest (ROI)
        roi = image[50:250, 50:250]  # Assuming similar 200x200 window for hand

        # Apply black background filter
        roi_filtered = apply_black_background(roi)

        # Predict the sign from the ROI with black background
        predicted_label = predict_sign(roi_filtered)

        # Update the label with the predicted sign
        word_label.config(text=f"Predicted Sign: {predicted_label}")

        # Convert the image (ROI with black background) to display in Tkinter
        roi_rgb = cv2.cvtColor(roi_filtered, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(roi_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        # Display the uploaded image on the canvas
        canvas_label.imgtk = imgtk
        canvas_label.configure(image=imgtk)

# Function to update the real-time webcam feed
def update_webcam_feed():
    global cap, live_feed_label

    # Capture the frame from webcam
    ret, frame = cap.read()
    if ret:
        # Draw a rectangle to show the ROI
        cv2.rectangle(frame, (50, 50), (250, 250), (255, 0, 0), 2)
        cv2.putText(frame, "Place hand here", (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Convert frame to RGB for displaying in Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)

        # Update the live webcam feed in the label
        live_feed_label.imgtk = imgtk
        live_feed_label.configure(image=imgtk)

    # Repeat the function after 10 milliseconds
    live_feed_label.after(10, update_webcam_feed)

# Function to automatically capture and predict every 2 seconds
def auto_predict():
    capture_snapshot()
    root.after(2000, auto_predict)  # Schedule the function to be called after 2 seconds

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Create the Tkinter window
root = Tk()
root.title("Sign Language Detection - Webcam Snapshot")

# Create a frame for the live feed and the prediction side-by-side
main_frame = Frame(root)
main_frame.pack()

# Create a label to display the real-time webcam feed
live_feed_label = Label(main_frame)
live_feed_label.grid(row=0, column=0)

# Create a frame for the prediction label and the snapshot
prediction_frame = Frame(main_frame)
prediction_frame.grid(row=0, column=1, padx=10)

# Create a label to display the predicted word
word_label = Label(prediction_frame, text="Predicted Sign: ", font=("Helvetica", 16))
word_label.pack(pady=10)

# Label to display the snapshot with the black background, placed below the predicted sign
canvas_label = Label(prediction_frame)
canvas_label.pack()

# Create a button to capture a snapshot and predict
capture_button = Button(root, text="Capture and Predict", command=capture_snapshot)
capture_button.pack(pady=10)

# Create a button to upload an image and predict
upload_button = Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=10)

# Start updating the real-time webcam feed
update_webcam_feed()

# Start automatic prediction every 2 seconds
auto_predict()

# Start the Tkinter main loop
root.mainloop()

# Release the webcam after closing the GUI
cap.release()
