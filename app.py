import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import pandas as pd
from utils import set_background, write_csv
import io
from predict import predict_characters

# Set the background image for the Streamlit app
set_background("./images/background.jpg")

# Define the directory paths for the pre-trained models
LICENSE_MODEL_DETECTION_DIR = './saved_models/license_plate_detector.pt'
COCO_MODEL_DIR = "./saved_models/yolov8n.pt"

# Define the vehicle class IDs that will be detected
vehicles = [2, 3, 5, 7]  # car, motorcycle, bus, and truck

# Create containers for header and body
header = st.container()
body = st.container()

# Load the pre-trained models
coco_model = YOLO(COCO_MODEL_DIR)
license_plate_detector = YOLO(LICENSE_MODEL_DETECTION_DIR)

# Set the detection threshold
threshold = 0.15

# Initialize the state of the application
state = "Uploader"
if "state" not in st.session_state:
    st.session_state["state"] = "Uploader"

# Define a function to perform model predictions
def model_prediction(img):
    license_numbers = 0
    results = {}
    licenses_texts = []
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Detect objects using the COCO model
    object_detections = coco_model(img)[0]
    # Detect license plates using the license plate detector model
    license_detections = license_plate_detector(img)[0]

    # Draw bounding boxes around detected vehicles
    if len(object_detections.boxes.cls.tolist()) != 0:
        for detection in object_detections.boxes.data.tolist():
            xcar1, ycar1, xcar2, ycar2, car_score, class_id = detection
            if int(class_id) in vehicles:
                cv2.rectangle(img, (int(xcar1), int(ycar1)), (int(xcar2), int(ycar2)), (0, 0, 255), 3)

    # Process the license plate detections
    if len(license_detections.boxes.cls.tolist()) != 0:
        license_plate_crops_total = []
        for license_plate in license_detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
            license_plate_crop = img[int(y1):int(y2), int(x1): int(x2), :]

            # Convert the license plate crop to an in-memory buffer
            is_success, buffer = cv2.imencode(".jpg", license_plate_crop)
            io_buf = io.BytesIO(buffer)

            # Use the predict_characters function from predict.py to extract the license plate text
            license_plate_text = predict_characters(io_buf.getvalue())
            licenses_texts.append(license_plate_text)

            if license_plate_text:
                license_plate_crops_total.append(license_plate_crop)
                results[license_numbers] = {}
                results[license_numbers][license_numbers] = {
                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2], 'car_score': car_score},
                    'license_plate': {'bbox': [x1, y1, x2, y2],
                                     'text': license_plate_text,
                                     'bbox_score': score}
                }
                license_numbers += 1

        # Write the detection results to a CSV file
        write_csv(results, f"./csv_detections/detection_results.csv")

        # Convert the image back to RGB format for display
        img_wth_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return [img_wth_box, licenses_texts, license_plate_crops_total]

    else:
        img_wth_box = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if len(object_detections.boxes.cls.tolist()) == 0:
            st.warning("No vehicle detected in the image.")
        elif len(license_detections.boxes.cls.tolist()) == 0:
            st.warning("No license plate detected in the vehicle.")
        return [img_wth_box]

# Functions to change the state of the application
def change_state_uploader():
    st.session_state["state"] = "Uploader"

def change_state_camera():
    st.session_state["state"] = "Camera"

# Streamlit UI and state management
with header:
    _, col1, _ = st.columns([0.2, 1, 0.1])
    col1.title("License Plate Detector")

    _, col2, _ = st.columns([0.05, 1, 0.1])
    st.write("The different models detect the car and the license plate in a given image, then extract the info about the license using CNN, and crop and save the license plate as an Image, with a CSV file with all the data.")

with body:
    _, col1, _ = st.columns([0.1, 1, 0.2])
    col1.subheader("Try It-out the License Car Plate Detection Model!")

    _, colb1, colb2, colb3 = st.columns([0.2, 0.7, 0.6, 1])

    if colb1.button("Upload a Quality Image", on_click=change_state_uploader):
        pass
    elif colb3.button("Take a Photo", on_click=change_state_camera):
        pass

    if st.session_state["state"] == "Uploader":
        img = st.file_uploader("Upload a Car Image: ", type=["png", "jpg", "jpeg"])
    elif st.session_state["state"] == "Camera":
        img = st.camera_input("Take a Photo: ")

    _, col2, _ = st.columns([0.3, 1, 0.2])
    _, col5, _ = st.columns([0.8, 1, 0.2])

    if img is not None:
        image = np.array(Image.open(img))
        col2.image(image, width=400)

        if col5.button("Apply Detection"):
            results = model_prediction(image)

            if len(results) == 3:
                prediction, texts, license_plate_crop = results[0], results[1], results[2]

                texts = [i for i in texts if i is not None]

                if len(texts) == 1 and len(license_plate_crop):
                    _, col3, _ = st.columns([0.4, 1, 0.2])
                    col3.header("Detection Results ✅:")

                    _, col4, _ = st.columns([0.1, 1, 0.1])
                    col4.image(prediction)

                    _, col9, _ = st.columns([0.4, 1, 0.2])
                    col9.header("License Cropped ✅:")

                    _, col10, _ = st.columns([0.3, 1, 0.1])
                    col10.image(license_plate_crop[0], width=350)

                    _, col11, _ = st.columns([0.45, 1, 0.55])
                    col11.success(f"License Number: {texts[0]}")

                    df = pd.read_csv(f"./csv_detections/detection_results.csv")
                    st.dataframe(df)
                elif len(texts) > 1 and len(license_plate_crop) > 1:
                    _, col3, _ = st.columns([0.4, 1, 0.2])
                    col3.header("Detection Results ✅:")

                    _, col4, _ = st.columns([0.1, 1, 0.1])
                    col4.image(prediction)

                    _, col9, _ = st.columns([0.4, 1, 0.2])
                    col9.header("License Cropped ✅:")

                    _, col10, _ = st.columns([0.3, 1, 0.1])
                    _, col11, _ = st.columns([0.45, 1, 0.55])

                    col7, col8 = st.columns([1, 1])
                    for i in range(0, len(license_plate_crop)):
                        col10.image(license_plate_crop[i], width=350)
                        col11.success(f"License Number {i}: {texts[i]}")

                    df = pd.read_csv(f"./csv_detections/detection_results.csv")
                    st.dataframe(df)
            else:
                prediction = results[0]
                _, col3, _ = st.columns([0.4, 1, 0.2])
                col3.header("Detection Results ✅:")

                _, col4, _ = st.columns([0.3, 1, 0.1])
                col4.image(prediction)