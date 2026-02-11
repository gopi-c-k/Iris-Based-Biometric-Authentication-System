import streamlit as st
import pickle
import cv2
import numpy as np
from skimage.feature import hog


with open("svm_iris_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)


eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

def detect_eye(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
    return len(eyes) > 0  

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    image = cv2.resize(image, (128, 128))  
    features, _ = hog(image, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True)
    return features


st.title("🔐 Iris Authentication System")
st.write("Upload your iris image to verify your identity and access the system.")


uploaded_file = st.file_uploader("Choose an iris image...", type=["png", "jpg", "jpeg", "bmp"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    if not detect_eye(image):
        st.error("Please upload a valid iris image.")
    else:
        
        features = preprocess_image(image)
        prediction = model.predict([features])

        
        if prediction[0]:
            st.success(f"✅ Authentication Successful! Welcome, Person {prediction[0]}!")
            st.balloons()
            st.subheader("🏠 Welcome to the System!")
            st.write("You have successfully logged in using iris authentication.")
        else:
            st.error("❌ Authentication Failed! Identity not recognized.")
