# Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import tensorflow as tf

# Loading the Model
model = load_model('plant_disease_model.h5')

# Name of Classes
CLASS_NAMES = ('Tomato-Bacterial_spot', 'Potato-Barly blight', 'Corn-Common_rust')

# Custom CSS for styling
st.markdown(
    '''
    <style>
        body {
            background-color: #F0F2F6;
        }
        h1 {
            color: #2E8B57;
            text-align: center;
        }
        .stButton>button {
            background-color: #2E8B57;
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1rem;
        }
    </style>
    ''',
    unsafe_allow_html=True
)

# Setting Title of App
st.title("🌿 Plant Disease Detection")
st.markdown("## Upload an image of the plant leaf and detect the disease")

# Uploading the plant image
plant_image = st.file_uploader("📷 Choose an image...", type=["jpg", "jpeg", "png", "JPG", "PNG", "JPEG"])
submit = st.button('🔍 Predict Disease')

# On predict button click
if submit:
    if plant_image is not None:
        # Convert the file to an OpenCV image
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        # Displaying the uploaded image
        st.image(opencv_image, channels="BGR", caption="Uploaded Image", use_column_width=True)

        # Resizing the image
        opencv_image = cv2.resize(opencv_image, (256, 256))

        # Convert image to 4D array
        opencv_image = opencv_image.reshape((1, 256, 256, 3))

        # Make Prediction
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]

        # Displaying the result
        st.success(f"🌿 This is a {result.split('-')[0]} leaf with {result.split('-')[1]}")
    else:
        st.warning("Please upload an image before proceeding.")
