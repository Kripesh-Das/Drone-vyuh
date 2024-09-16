from flask import Flask, render_template, request
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import gdown
import os

app = Flask(__name__)

# Function to download the model from Google Drive
def download_model_from_drive(file_id, output_path):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_path, quiet=False)

# Load the model (download if not exists)
def load_model():
    model_path = 'VGG_19_model.h5'
    drive_file_id = '1mnTChV2-KcALBvIU0yw63ISs3nPX4lq0'  # Replace with your actual file ID
    if not os.path.exists(model_path):
        download_model_from_drive(drive_file_id, model_path)
    return tf.keras.models.load_model(model_path, custom_objects={'ExponentialDecay': tf.keras.optimizers.schedules.ExponentialDecay})

model = load_model()

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).resize((224, 224))
    img_array = np.expand_dims(np.array(img), axis=0)
    img_array = tf.keras.applications.vgg19.preprocess_input(img_array)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        file = request.files['image']
        image_bytes = file.read()
        img_array = preprocess_image(image_bytes)
        predictions = model.predict(img_array)
        output = np.argmax(predictions, axis=1)[0]
        result_map = {0: "🌀 Long Blades Rotor", 1: "🌀 Short Blade Rotor", 2: "🦜 Bird", 3: "🦜 Bird + 2 Blade Rotor", 4: "✈️ RC Plane", 5: "🚁 Drone"}
        result = result_map.get(output, "🔍 Unknown Prediction")
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
