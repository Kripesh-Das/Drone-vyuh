import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image
import io
import base64

# Load the VGG-19 model from the .h5 file with custom objects
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        'VGG_19_model.h5',
        custom_objects={'ExponentialDecay': tf.keras.optimizers.schedules.ExponentialDecay}
    )

model = load_model()

def preprocess_image(image_bytes):
    """Preprocess the image to the format that the model expects."""
    img = Image.open(io.BytesIO(image_bytes)).resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = tf.keras.applications.vgg19.preprocess_input(img_array)  # Preprocess as per VGG19 requirements
    return img_array

# Define the Streamlit app
# st.set_page_config(page_title="AI Image Classifier", layout="wide")

# Apply custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #333;
        color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 12px 24px;
        border-radius: 30px;
        font-size: 18px;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
    .image-container {
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 20px;
        margin-top: 30px;
    }
    .image-container img {
        max-width: 100%;
        max-height: 400px;
        border-radius: 10px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    .image-container img:hover {
        transform: scale(1.05);
    }
    .prediction-result {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        margin-top: 20px;
        padding: 20px;
        background-color: #e8f5e9;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: #333;
    }
    .footer {
        text-align: center;
        margin-top: 50px;
        color: #888;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 AeroGuard AI")
st.write("Upload an image to classify it's drone or bird.")

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    img = Image.open(io.BytesIO(image_bytes)).resize((224, 224))
    img_array = preprocess_image(image_bytes)
    
    # Predict using the model
    with st.spinner('Analyzing image...'):
        predictions = model.predict(img_array)
    output = np.argmax(predictions, axis=1).tolist()[0]
    
    # Map the prediction to specific rotor types or items
    result_map = {
        0: "🌀 Long Blades Rotor",
        1: "🌀 Short Blade Rotor",
        2: "🦜 Bird",
        3: "🦜 Bird + 2 Blade Rotor",
        4: "✈️ RC Plane",
        5: "🚁 Drone"
    }
    result = result_map.get(output, "🔍 Unknown Prediction")
    
    # Display uploaded image and prediction result
    st.markdown("""
        <div class="image-container">
            <img src="data:image/jpeg;base64,{}" alt="Uploaded Image" />
        </div>
    """.format(base64.b64encode(image_bytes).decode()), unsafe_allow_html=True)
    
    st.markdown(f'<div class="prediction-result">{result}</div>', unsafe_allow_html=True)

# Download Images Section
st.subheader("📂 Sample Images")
st.write("Click the button below to access and download sample images for testing.")

# Link to Google Drive folder
drive_link = "https://drive.google.com/drive/folders/1rlncYHJT8I6QDgzdrJLR83qi-SXMiizc?usp=sharing"

# Display a button to open the Google Drive link
if st.button('Access Sample Images'):
    st.markdown(f"[Open Google Drive Folder]({drive_link})", unsafe_allow_html=True)

# Add a footer
st.markdown("""
    <div class="footer">
        <p>Powered by VGG-19 | Developed with ❤️ by ???</p>
    </div>
""", unsafe_allow_html=True)
