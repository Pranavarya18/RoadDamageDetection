import logging
import queue
from pathlib import Path
from typing import List, NamedTuple

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# Deep learning framework
from ultralytics import YOLO

# Logger setup
logger = logging.getLogger(__name__)

# STUN Server for WebRTC connection
from sample_utils.get_STUNServer import getSTUNServer
STUN_STRING = "stun:" + str(getSTUNServer())
STUN_SERVER = [{"urls": [STUN_STRING]}]

# Streamlit page configuration
st.set_page_config(
    page_title="Realtime Road Damage Detection",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load the YOLO model (session caching)
cache_key = "yolov8smallrdd"
if cache_key in st.session_state:
    net = st.session_state[cache_key]
else:
    # Use your locally saved YOLOv8 model
    MODEL_LOCAL_PATH = Path('./models/YOLOv8_Small_RDD.pt')  # Ensure this is the correct local path
    net = YOLO(MODEL_LOCAL_PATH)
    st.session_state[cache_key] = net

# Class names for damage detection
CLASSES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Potholes"
]

# NamedTuple for detection results
class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray

# Title and description
st.title("Realtime Road Damage Detection")
st.write("Detect road damage in real-time using your webcam. This app detects various types of road damage such as cracks and potholes using the YOLOv8 model trained on the CRDDC2022 dataset.")

# Create queue for storing detection results
result_queue: "queue.Queue[List[Detection]]" = queue.Queue()

# Video frame callback function to process each frame
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    h_ori = image.shape[0]
    w_ori = image.shape[1]
    
    # Resize image for model inference
    image_resized = cv2.resize(image, (640, 640), interpolation=cv2.INTER_AREA)
    results = net.predict(image_resized, conf=score_threshold)
    
    # Collect detections in queue
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
        result_queue.put(detections)

    # Annotate frame with results
    annotated_frame = results[0].plot()
    _image = cv2.resize(annotated_frame, (w_ori, h_ori), interpolation=cv2.INTER_AREA)
    
    return av.VideoFrame.from_ndarray(_image, format="bgr24")

# Setup WebRTC context for real-time video streaming
webrtc_ctx = webrtc_streamer(
    key="road-damage-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": STUN_SERVER},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": {"width": {"ideal": 1280, "min": 800}},
        "audio": False
    },
    async_processing=True,
)

# Confidence threshold slider
score_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
st.write("Lower the threshold if no damage is detected, and increase it if false predictions occur.")

st.divider()

# Option to display the predictions in table format
if st.checkbox("Show Predictions Table", value=False):
    if webrtc_ctx.state.playing:
        labels_placeholder = st.empty()
        while True:
            result = result_queue.get()
            labels_placeholder.table(result)
