import streamlit as st
from ultralytics import YOLO
import numpy as np
import cv2
import pandas as pd
import time
from pathlib import Path

# === Page Setup ===
st.set_page_config(
    page_title="DTU Aqua Multitool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Minimal & Modern Headline ---
st.markdown(
    """
    <div style='text-align: center; margin-top: 2.5em; margin-bottom: 0.7em;'>
        <h1 style='font-size: 2.5em; font-weight: 700; color: #183153; margin-bottom: 0.15em; letter-spacing: 1px; font-family: "Segoe UI", "Arial", sans-serif;'>DTU Aqua Multitool</h1>
        <div style='width: 80px; height: 4px; background: #00b4d8; margin: 0.5em auto 1.2em auto; border-radius: 2px;'></div>
        <h4 style='font-weight: 400; color: #4a6fa5; margin-top: 0; margin-bottom: 0.2em; font-size: 1.15em;'>Advanced Fish Detection & Measurement Platform</h4>
    </div>
    <div style='text-align: center; margin-bottom: 1.5em;'>
        <img src='https://openmoji.org/data/color/svg/1F41F.svg' alt='Fish1' height='38' style='margin:0 10px; vertical-align:middle;'/>
        <img src='https://openmoji.org/data/color/svg/1F420.svg' alt='Fish2' height='38' style='margin:0 10px; vertical-align:middle;'/>
        <img src='https://openmoji.org/data/color/svg/1F988.svg' alt='Fish3' height='38' style='margin:0 10px; vertical-align:middle;'/>
    </div>
    """,
    unsafe_allow_html=True
)

# --- App Description with Spacing ---
st.markdown(
    """
    <div style='text-align: center; margin-bottom: 2.2em; margin-top: 0.5em;'>
        <span style='font-size: 1.15em; color: #333; background: #f3f8fa; border-radius: 8px; padding: 0.7em 2em; display: inline-block;'>
            Fish detection and measurement with real-time model integration.
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

# === Load Model ===
@st.cache_resource
def load_model():
    model_path = Path("models/yolo11m-seg.pt")
    if not model_path.exists():
        st.error("Model file not found. Please ensure 'models/yolo11m-seg.pt' exists in the project directory.")
        st.stop()
    return YOLO(str(model_path))

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# === Sidebar Controls ===
with st.sidebar:
    st.header("Settings")
    confidence_threshold = st.slider(
        "Detection Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        help="Adjust the confidence threshold for detection"
    )
    
    mode = st.radio(
        "Operation Mode:",
        ["Batch Processing", "Field Mode", "Live Camera"],
        help="Select how you want to process images"
    )

# === Mode Descriptions ===
mode_descriptions = {
    "Batch Processing": "<b>Batch Processing</b>: Upload and analyze multiple images or videos at once. Ideal for processing large datasets.",
    "Field Mode": "<b>Field Mode</b>: Designed for quick, single-image analysis in field conditions. Upload one image and get instant results.",
    "Live Camera": "<b>Live Camera</b>: Use your device's camera for real-time fish detection and measurement. Great for live demonstrations or on-site analysis."
}
st.markdown(
    f"""
    <div style='margin-bottom:2.2em; margin-top: 0.2em;'>
        <span style='background: #eaf4fb; color:#005691; border-radius: 8px; padding: 0.7em 1.5em; display: inline-block; font-size: 1.08em;'>
            {mode_descriptions[mode]}
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

# === Live Camera Stream ===
if mode == "Live Camera":
    st.markdown("### Live Camera Stream")
    
    # Camera selection
    try:
        available_cameras = []
        for i in range(3):  # Check first 3 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        
        if not available_cameras:
            st.error("No cameras detected!")
            st.stop()
            
        camera_idx = st.selectbox("Select Camera", available_cameras)
    except Exception as e:
        st.error(f"Error accessing cameras: {str(e)}")
        st.stop()

    # Stream controls
    col1, col2 = st.columns(2)
    with col1:
        start_stream = st.button("Start Stream")
    with col2:
        stop_stream = st.button("Stop Stream")

    # Stream placeholder
    frame_placeholder = st.empty()
    
    if start_stream:
        cap = cv2.VideoCapture(camera_idx)
        
        while cap.isOpened() and not stop_stream:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to get camera stream.")
                break
                
            # Process frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(
                source=frame_rgb,
                task="segment",
                conf=confidence_threshold,
                verbose=False
            )
            
            # Display results
            if results and len(results) > 0:
                annotated = results[0].plot()
                frame_placeholder.image(annotated, channels="RGB", use_container_width=True)
            
            time.sleep(0.1)  # Prevent excessive CPU usage
            
        cap.release()

else:
    # === File Processing ===
    uploaded = st.file_uploader(
        "Upload image or video",
        type=["jpg", "jpeg", "png", "mp4", "avi"],
        help="Supported formats: JPG, PNG, MP4, AVI"
    )

    if uploaded:
        st.markdown("### Preview")
        st.image(uploaded, use_container_width=True)
        
        if st.button("Detect Objects"):
            with st.spinner("Processing..."):
                try:
                    # Convert uploaded file to numpy array
                    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
                    
                    if uploaded.type.startswith('image'):
                        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                        if image is None:
                            st.error("Failed to read image file.")
                            st.stop()
                    else:  # Video file
                        st.warning("Video processing is currently optimized for the first frame only.")
                        vid = cv2.VideoCapture(uploaded.name)
                        ret, image = vid.read()
                        vid.release()
                        if not ret:
                            st.error("Failed to read video file.")
                            st.stop()
                    
                    # Process image
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = model.predict(
                        source=image_rgb,
                        task="segment",
                        conf=confidence_threshold,
                        verbose=False
                    )
                    
                    if results and len(results) > 0:
                        r = results[0]
                        
                        # Display results
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### Detected Objects")
                            annotated = r.plot()
                            st.image(annotated, channels="RGB", use_container_width=True)
                        
                        with col2:
                            st.markdown("### Results Table")
                            # Create results dataframe
                            data = []
                            for box, score, cls in zip(r.boxes.xyxy.tolist(),
                                                     r.boxes.conf.tolist(),
                                                     r.boxes.cls.tolist()):
                                data.append({
                                    "Class": int(cls),
                                    "Confidence": f"{float(score):.2f}",
                                    "X1": int(box[0]),
                                    "Y1": int(box[1]),
                                    "X2": int(box[2]),
                                    "Y2": int(box[3])
                                })
                            
                            if data:
                                df = pd.DataFrame(data)
                                st.dataframe(df, use_container_width=True)
                                
                                # Download results
                                csv = df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    "Download Results (CSV)",
                                    data=csv,
                                    file_name="detection_results.csv",
                                    mime="text/csv"
                                )
                            else:
                                st.info("No objects detected in the image.")
                    else:
                        st.warning("No detections found.")
                        
                except Exception as e:
                    st.error(f"An error occurred during processing: {str(e)}")

# === Footer ===
st.markdown("---")
with st.expander("About"):
    st.markdown("""
    ### DTU Aqua Multitool
    A powerful tool for fish detection and measurement using state-of-the-art computer vision.
    
    - Supports multiple input modes
    - Real-time object detection and segmentation
    - Adjustable detection confidence
    - Export results in CSV format
    """)
st.caption("© 2025 DTU Aqua Multitool | Version 1.0")
