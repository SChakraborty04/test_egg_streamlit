import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import os
from PIL import Image
import io

# Configuration
model_path = 'egg_yolov8.pt'  # Path to trained model (adjust for deployment)
conf_threshold = 0.5
iou_threshold = 0.45
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

# Streamlit app title and description
st.title("Egg Counter App")
st.write("Upload an image to count the number of eggs using a trained YOLOv8 model. The app will display the image with bounding boxes around detected eggs and show the total count.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load image
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB) if image_np.shape[2] == 4 else cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Processing image...")

        # Load YOLOv8 model
        try:
            model = YOLO(model_path)
            st.write("Model loaded successfully")
        except Exception as e:
            st.error(f"Error loading model: {e}")
            st.stop()

        # Run inference
        try:
            results = model(image_rgb, conf=conf_threshold, iou=iou_threshold, imgsz=640, device='cpu')  # Use CPU for simplicity; adjust for GPU
        except Exception as e:
            st.error(f"Error during inference: {e}")
            st.stop()

        # Count eggs and draw bounding boxes
        eggs_counter = 0
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
        for result in results:
            for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
                if model.names[int(cls)] == 'Egg':
                    eggs_counter += 1
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(image_bgr, f"Egg: {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw egg count
        cv2.rectangle(image_bgr, (image_bgr.shape[1] - 70, 215), (image_bgr.shape[1] - 5, 270), (0, 255, 0), 2)
        cv2.putText(image_bgr, str(eggs_counter), (image_bgr.shape[1] - 55, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        # Convert back to RGB for display
        image_display = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Save output image
        output_path = os.path.join(output_dir, 'processed_egg_count.jpg')
        cv2.imwrite(output_path, image_bgr)

        # Display results
        st.image(image_display, caption=f"Processed Image (Eggs Counted: {eggs_counter})", use_column_width=True)
        st.success(f"Total eggs counted: {eggs_counter}")

        # Provide download button for output image
        with open(output_path, "rb") as file:
            btn = st.download_button(
                label="Download Processed Image",
                data=file,
                file_name="processed_egg_count.jpg",
                mime="image/jpeg"
            )

    except Exception as e:
        st.error(f"Error processing image: {e}")

else:
    st.info("Please upload an image to start.")