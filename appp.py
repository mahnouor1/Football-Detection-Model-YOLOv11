import cv2
import streamlit as st
from pathlib import Path
import os
from ultralytics import YOLO
from PIL import Image
import time
import numpy as np
import tempfile

# --- Page Configuration ---
st.set_page_config(page_title="Football Detector", page_icon="⚽")
st.header("Football Detection using YOLO ⚽")

# --- Sidebar Model Configuration ---
st.sidebar.header("Model Configurations")
confidence_value = float(st.sidebar.slider("Select Model Confidence Value", 25, 100, 40)) / 100

# --- Load YOLO Model ---
@st.cache_resource
def load_model(model_path):
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Unable to load model from: {model_path}")
        st.error(e)
        return None

# Define your model file path (adjust as needed)
ROOT = Path(os.getcwd())
MODEL_DIR = ROOT / 'weight'
DETECTION_MODEL = MODEL_DIR / 'bestt.pt'
if not DETECTION_MODEL.exists():
    st.error(f"Model file not found at: {DETECTION_MODEL}")
    st.stop()  # Stop the app if the model file is missing

model = load_model(DETECTION_MODEL)

# --- Source Selection ---
source_radio = st.sidebar.radio("Select Source", ["Image", "Video"])

# ===============================
# Process Uploaded Image
# ===============================
if source_radio == "Image":
    uploaded_image = st.sidebar.file_uploader("Upload an Image", type=["jpg", "png", "jpeg", "bmp", "webp"])
    
    if uploaded_image is not None:
        try:
            # Open and display the uploaded image
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            # When user clicks on the detect button, process the image
            if st.sidebar.button("Detect Objects in Image"):
                with st.spinner("Processing image..."):
                    # YOLO's predict method works with PIL images or numpy arrays
                    result = model.predict(image, conf=confidence_value)
                    # Extract detection boxes and plotted image
                    boxes = result[0].boxes
                    # The plot() function returns a BGR image; convert it to RGB
                    result_plotted = cv2.cvtColor(result[0].plot(), cv2.COLOR_BGR2RGB)
                    
                    # Display the detected image
                    st.image(result_plotted, caption="Detected Image", use_container_width=True)
                    
                    # Provide a download button for the detected image
                    st.download_button(
                        label="Download Detected Image",
                        data=cv2.imencode(".jpg", result_plotted)[1].tobytes(),
                        file_name="detected_image.jpg",
                        mime="image/jpeg"
                    )
                    
                    # Optionally, show detection results (e.g., bounding box coordinates)
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
        except Exception as e:
            st.error("Error processing the image:")
            st.error(e)
    else:
        st.info("Please upload an image to begin.")

# ===============================
# Process Uploaded Video
# ===============================
elif source_radio == "Video":
    uploaded_video = st.sidebar.file_uploader("Upload a Video", type=["mp4", "mov", "avi", "mkv"])
    
    if uploaded_video is not None:
        try:
            # Save the uploaded video to a temporary file so OpenCV can read it
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            tfile.close()
            
            # Display the uploaded video
            st.video(tfile.name)
            
            if st.sidebar.button("Detect Objects in Video"):
                st_frame = st.empty()      # For dynamically updating video frames
                progress_bar = st.progress(0)
                
                def process_video(video_path):
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        st.error("Error: Could not open video.")
                        return
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    current_frame = 0
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        current_frame += 1
                        yield current_frame, total_frames, frame
                    cap.release()
                
                # Process each frame and display the detection results
                for current_frame, total_frames, frame in process_video(tfile.name):
                    progress_bar.progress(current_frame / total_frames)
                    # Optionally, resize the frame for faster processing or to fit the UI
                    frame_resized = cv2.resize(frame, (720, int(720 * (9 / 16))))
                    result = model.predict(frame_resized, conf=confidence_value)
                    result_plotted = result[0].plot()
                    st_frame.image(result_plotted, caption="Detected Video Frame", channels="BGR", use_container_width=True)
                    time.sleep(0.03)  # Adjust delay for playback speed
        except Exception as e:
            st.error("Error processing the video:")
            st.error(e)
    else:
        st.info("Please upload a video to begin.")

# --- Clear Cache Button ---
if st.sidebar.button("Clear Cache"):
    st.cache_resource.clear()
    st.success("Cache cleared successfully!")
