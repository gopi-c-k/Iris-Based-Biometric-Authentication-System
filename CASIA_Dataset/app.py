import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

MODEL_PATH = "mobilenetv2_casia.pkl"
NUM_CLASSES = 108  
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), 
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


@st.cache_resource
def load_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.last_channel, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


model = load_model()


def is_eye_image(img):
    img_np = np.array(img)

    edges = cv2.Canny(img_np, threshold1=100, threshold2=200)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
                if len(approx) > 5: 
                    return True 

    return False  


st.title("🔍 Iris Recognition - CASIA Model")
st.caption("Upload a **segmented grayscale iris image**. The model predicts the Person ID using MobileNetV2.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(input_tensor)
        _, prediction = torch.max(output, 1)

    st.success(f"✅ Predicted Person ID: `{prediction.item()}`")
