# Road Damage Detection Application

## Overview

The **Road Damage Detection Application** is designed to enhance road safety and infrastructure maintenance using deep learning. This application uses the **YOLOv8** model, which has been trained on the **Crowdsensing-based Road Damage Detection Challenge 2022 (CRDDC2022)** dataset.

The model detects various types of road damage, such as cracks and potholes, and categorizes them into specific classes.

## Features

- **Real-time Road Damage Detection**: Detect road damage instantly using webcam or video input.
- **Types of Road Damage Detected**:
    - **Longitudinal Crack**
    - **Transverse Crack**
    - **Alligator Crack**
    - **Potholes**
  
- **Supported Input Methods**:
    - Real-time webcam feed
    - Video uploads
    - Image uploads

- **Fast Detection**: The application leverages YOLOv8 Small, a lightweight and fast deep learning model trained on the **Japan, India, China, Czech, United States** datasets, excluding **Norway**.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Pranavarya18/RoadDamageDetection.git
    cd RoadDamageDetection
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the pre-trained YOLOv8 model:
    ```bash
    wget https://github.com/Pranavarya18/RoadDamageDetection/raw/main/models/YOLOv8_Small_RDD.pt -P ./models
    ```

4. Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

## Usage

### Real-time Detection (Webcam)
- Select the **Webcam** input option to detect road damage in real-time.
- The application will show a live webcam feed and highlight detected road damage with bounding boxes.

### Image Detection
- Upload an image to detect road damage.
- The model will predict the damage type and display the results along with confidence scores.

### Video Detection
- Upload a video for batch processing.
- The application will analyze the video, frame by frame, and highlight detected road damages in each frame.

## Model Details

- **Model**: YOLOv8 Small
- **Dataset**: Crowdsensing-based Road Damage Detection Challenge 2022 (CRDDC2022)
- **Training Data**: The model was trained on road damage datasets from **Japan, India, China, Czech, United States**.
- **Damage Types**: Longitudinal Crack, Transverse Crack, Alligator Crack, and Potholes.

## How It Works

1. **Pre-processing**: Images or videos are resized to a consistent shape (640x640) for the model.
2. **Prediction**: The model performs object detection on each frame/image, categorizing the detected road damage types.
3. **Post-processing**: Detected bounding boxes are drawn on the image or video frame, highlighting the detected damages.
4. **Result Display**: The result is displayed back to the user with bounding boxes drawn on the original image/video.

## Contact

For inquiries or feedback, feel free to reach out:
- GitHub: [RoadDamageDetection](https://github.com/Pranavarya18/RoadDamageDetection)

---

**Note**: Replace any placeholders with specific URLs or paths as required. Also, feel free to add or remove sections based on your project's specific needs.
