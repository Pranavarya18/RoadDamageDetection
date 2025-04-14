import streamlit as st

st.set_page_config(
    page_title="Road Damage Detection App",
    page_icon="🛣️",
)

# Application Title
st.title("Road Damage Detection Application")

# Description and Overview of the Project
st.markdown(
    """
    Welcome to the Road Damage Detection Application, developed to enhance road safety and infrastructure maintenance by leveraging deep learning. This application uses the **YOLOv8** model, which has been trained on the **Crowdsensing-based Road Damage Detection Challenge 2022 (CRDDC2022)** dataset.

    **Features:**
    - **Real-time Road Damage Detection**: Detect various types of road damage including potholes and cracks.
    - **Types of Damage Detected**:
        - Longitudinal Crack
        - Transverse Crack
        - Alligator Crack
        - Potholes

    The model is trained on the **YOLOv8 Small model** using the **Japan, India, China, Czech, United States, and other CRDDC2022 datasets** (excluding Norway) to provide accurate and swift road damage detection.

    **App Use Cases**:
    You can upload images, videos, or even use real-time webcam input to detect road damages on the go.
    """
)

# Divider
st.divider()
