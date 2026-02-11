import os
import json
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image

# ================= PATHS =================
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "trained_model", "plant_disease_prediction_model.h5")
class_indices_path = os.path.join(working_dir, "class_indices.json")

# ================= LOAD MODEL =================
model = tf.keras.models.load_model(model_path)

# load class indices
class_indices = json.load(open(class_indices_path))

# reverse mapping (index → class name)
class_labels = {v: k for k, v in class_indices.items()}


# ================= IMAGE PREPROCESS =================
def load_and_preprocess_image(img, target_size=(224, 224)):
    img = img.convert("RGB")
    img = img.resize(target_size)
    img = np.array(img)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img


# ================= STREAMLIT APP =================
st.set_page_config(page_title="Plant Disease Classifier", layout="centered")
st.title("🌿 Plant Disease Classifier")

uploaded_image = st.file_uploader(
    "Upload an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:
    image = Image.open(uploaded_image)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image.resize((180, 180)), caption="Uploaded Image")

    with col2:
        if st.button("Classify"):
            processed_image = load_and_preprocess_image(image)
            predictions = model.predict(processed_image)
            predicted_index = np.argmax(predictions)
            predicted_label = class_labels[predicted_index]

            st.success(f"Prediction: {predicted_label}")


