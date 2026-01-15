Road Damage Detection Application
📌 Overview
The Road Damage Detection Application is a deep learning–based system designed to improve road safety and infrastructure maintenance. It uses a YOLOv8 Small object detection model trained on the Crowdsensing-based Road Damage Detection Challenge 2022 (CRDDC2022) dataset to detect and classify different types of road damage such as cracks and potholes.

The application provides real-time and batch detection capabilities through an interactive Streamlit interface.

✨ Features
🚦 Real-time Road Damage Detection using webcam input

📷 Image-based Detection for uploaded images

🎥 Video-based Detection with frame-by-frame analysis

⚡ Fast and Lightweight Model using YOLOv8 Small

🧠 Multi-class Damage Classification with confidence scores

🧱 Types of Road Damage Detected
Longitudinal Crack

Transverse Crack

Alligator Crack

Potholes

🛠️ Supported Input Methods
Live webcam feed

Image upload

Video upload

🧠 Model Details
Model Architecture: YOLOv8 Small

Dataset: Crowdsensing-based Road Damage Detection Challenge 2022 (CRDDC2022)

Training Data Regions: Japan, India, China, Czech Republic, United States

Excluded Region: Norway

Input Size: 640 × 640

⚙️ Installation
1️⃣ Clone the Repository
bash
Copy code
git clone https://github.com/Pranavarya18/RoadDamageDetection.git
cd RoadDamageDetection
2️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
3️⃣ Download Pre-trained Model
bash
Copy code
wget https://github.com/Pranavarya18/RoadDamageDetection/raw/main/models/YOLOv8_Small_RDD.pt -P ./models
4️⃣ Run the Application
bash
Copy code
streamlit run app.py
▶️ Usage
📸 Real-time Detection (Webcam)
Select the webcam option in the application

Live feed is displayed with detected road damages highlighted using bounding boxes

🖼️ Image Detection
Upload an image

Detected road damage types are shown along with confidence scores

🎬 Video Detection
Upload a video file

The application processes each frame and highlights detected road damages

🔍 How It Works
Pre-processing

Input images or video frames are resized to 640×640

Prediction

YOLOv8 performs object detection on each frame

Post-processing

Bounding boxes and labels are drawn on detected damage areas

Result Display

Output is shown in the Streamlit interface with visual annotations

🚀 Applications
Smart city infrastructure monitoring

Road maintenance planning

Traffic safety analysis

Autonomous vehicle perception support

📬 Contact
For queries or feedback:

GitHub: https://github.com/Pranavarya18

⭐ Acknowledgements
YOLOv8 by Ultralytics

CRDDC2022 Dataset contributors
