import streamlit as st
import pickle
import cv2
import numpy as np
from skimage.feature import hog

# Load the trained SVM model
with open("svm_iris_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load Haar Cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Function to detect an eye in the image
def detect_eye(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
    return len(eyes) > 0  # Returns True if at least one eye is detected

# Function to preprocess the image and extract HOG features
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    image = cv2.resize(image, (128, 128))  # Resize for consistency
    features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True)
    return features

# Streamlit UI
st.title("🔐 Iris Authentication System")
st.write("Upload your iris image to verify your identity and access the system.")

# File uploader
uploaded_file = st.file_uploader("Choose an iris image...", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Check if the image contains an eye
    if not detect_eye(image):
        st.error("Please upload a valid iris image.")
    else:
        # Preprocess and predict
        features = preprocess_image(image)
        prediction = model.predict([features])

        # Authentication success screen
        if prediction[0]:
            st.success(f"✅ Authentication Successful! Welcome, Person {prediction[0]}!")
            st.balloons()
            st.subheader("🏠 Welcome to the System!")
            st.write("You have successfully logged in using iris authentication.")
        else:
            st.error("❌ Authentication Failed! Identity not recognized.")
