import os
import logging
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import streamlit as st

# Deep learning framework
from ultralytics import YOLO
from PIL import Image
from io import BytesIO

st.set_page_config(
    page_title="Image Detection",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

# Local model path (Ensure the YOLOv8 model is stored in the './models' folder)
MODEL_LOCAL_PATH = ROOT / "./models/YOLOv8_Small_RDD.pt"  # Replace with your correct model path

# Session-specific caching
# Load the model
cache_key = "yolov8smallrdd"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    net = YOLO(MODEL_LOCAL_PATH)
    st.session_state[cache_key] = net

CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Potholes"
]

class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray

st.title("Road Damage Detection - Image")
st.write("Detect road damage in an image input. Upload the image and start detecting. This section can be useful for examining baseline data.")

# Upload image from user
image_file = st.file_uploader("Upload Image", type=['png', 'jpg'])

# Slider for confidence threshold
score_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
st.write("Lower the threshold if there is no damage detected, and increase the threshold if there is false prediction.")

# Process and display the image if uploaded
if image_file is not None:

    # Load the image
    image = Image.open(image_file)
    
    # Create two columns for displaying the original and predicted images
    col1, col2 = st.columns(2)

    # Perform inference on the image
    _image = np.array(image)
    h_ori = _image.shape[0]
    w_ori = _image.shape[1]

    # Resize image for model inference
    image_resized = cv2.resize(_image, (640, 640), interpolation=cv2.INTER_AREA)
    results = net.predict(image_resized, conf=score_threshold)
    
    # Process results into detections
    for result in results:
        boxes = result.boxes.cpu().numpy()
        detections = [
            Detection(
                class_id=int(_box.cls),
                label=CLASSES[int(_box.cls)],
                score=float(_box.conf),
                box=_box.xyxy[0].astype(int),
            )
            for _box in boxes
        ]

    # Annotate the frame with bounding boxes and class labels
    annotated_frame = results[0].plot()
    _image_pred = cv2.resize(annotated_frame, (w_ori, h_ori), interpolation=cv2.INTER_AREA)

    # Display the original image
    with col1:
        st.write("#### Original Image")
        st.image(_image)
    
    # Display the image with predicted bounding boxes
    with col2:
        st.write("#### Predictions")
        st.image(_image_pred)

        # Create a download button for the predicted image
        buffer = BytesIO()
        _download_images = Image.fromarray(_image_pred)
        _download_images.save(buffer, format="PNG")
        _download_images_byte = buffer.getvalue()

        st.download_button(
            label="Download Prediction Image",
            data=_download_images_byte,
            file_name="RDD_Prediction.png",
            mime="image/png"
        )
