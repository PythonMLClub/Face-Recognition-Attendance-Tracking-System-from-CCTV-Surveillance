# pip install -r requirements.txt
# pip freeze > requirements.txt  
# streamlit run .\face_register.py

import streamlit as st
from utils.db_handler import insert_employee
import cv2
import numpy as np
import os

# Get the absolute path of the directory containing face_registration.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Page configuration
st.set_page_config(page_title="Employee Face Registration", layout="centered")
st.title("👤 Employee Face Registration")

# Input fields for employee details
emp_id = st.text_input("🔢 Employee ID")
full_name = st.text_input("👤 Full Name")
department = st.text_input("🏢 Department")

# Shift time dropdown
shift_options = [
    "09:00 to 18:00", "09:30 to 18:30", "10:00 to 19:00",
    "10:30 to 19:30", "11:00 to 20:00", "11:30 to 20:30"
]
shift_time = st.selectbox("⏰ Shift Time", shift_options)

# Option to upload image or capture via webcam
image_source = st.radio("📸 Select Image Source", ("Upload Photo", "Capture via Webcam"))

image_bytes = None
if image_source == "Upload Photo":
    uploaded_file = st.file_uploader("Upload Face Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_bytes = uploaded_file.read()
elif image_source == "Capture via Webcam":
    picture = st.camera_input("Take a picture")
    if picture:
        image_bytes = picture.getvalue()

# Register button
if st.button("✅ Register"):
    if emp_id and full_name and department and shift_time and image_bytes:
        try:
            # Skip saving the .bin file locally as per your request
            # The image_bytes will still be saved in the database via insert_employee

            # Save the image as .jpg for visual verification in the images folder
            images_dir = os.path.join(BASE_DIR, "images")
            os.makedirs(images_dir, exist_ok=True)
            image_path = os.path.join(images_dir, f"employee_{emp_id}.jpg")
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is not None:
                try:
                    cv2.imwrite(image_path, image)
                    if os.path.exists(image_path):
                        st.success(f"Image saved locally as {image_path} for visual verification")
                    else:
                        st.error(f"Failed to verify saved .jpg file at {image_path}")
                        raise FileNotFoundError(f"File {image_path} was not created")
                except Exception as e:
                    st.error(f"Failed to save .jpg file: {e}")
                    raise
            else:
                st.warning("Failed to decode image for saving as .jpg")

            # Convert emp_id to integer
            emp_id = int(emp_id)
            success, message = insert_employee(emp_id, full_name, department, shift_time, image_bytes, None)
            if success:
                st.success(message)
            else:
                st.error(message)
        except ValueError:
            st.error("Employee ID must be a valid integer.")
        except Exception as e:
            st.error(f"Registration error: {e}")
    else:
        st.warning("⚠️ Please fill all fields and provide an image.")