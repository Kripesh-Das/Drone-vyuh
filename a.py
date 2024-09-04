import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img 
import numpy as np
from PIL import Image
import io

# Load the VGG-19 model from the .h5 file with custom objects
model = tf.keras.models.load_model(
    'VGG_19_model.h5',
    custom_objects={'ExponentialDecay': tf.keras.optimizers.schedules.ExponentialDecay}
)

def preprocess_image(image_bytes):
    """Preprocess the image to the format that the model expects."""
    img = Image.open(io.BytesIO(image_bytes)).resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = tf.keras.applications.vgg19.preprocess_input(img_array)  # Preprocess as per VGG19 requirements
    return img_array

# Define the Streamlit app
st.title("Image Classification with VGG-19")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
    
    # Preprocess the image
    img_array = preprocess_image(image_bytes)
    
    # Predict using the model
    predictions = model.predict(img_array)
    output = np.argmax(predictions, axis=1).tolist()[0]

    # Map the prediction to specific rotor types or items
    if output == 0:
        result = "long_blades_rotor"
    elif output == 1:
        result = "short_blade_rotor"
    elif output == 2:
        result = "Bird"
    elif output == 3:
        result = "Bird+2_Blade_rotor"
    elif output == 4:
        result = "RC plane"
    elif output == 5:
        result = "drone"
    else:
        result = "Unknown prediction"

    st.write("Prediction:", result)

# Download images section
st.subheader("Download Sample Images")

# Link to Google Drive folder
drive_link = "https://drive.google.com/drive/folders/1rlncYHJT8I6QDgzdrJLR83qi-SXMiizc?usp=sharing"  # Replace with your folder link

st.write("You can download sample images from the following link:")
st.markdown(f"[Download Sample Images from Google Drive]({drive_link})")
