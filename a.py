from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the VGG-19 model
def load_model():
    try:
        model = tf.keras.models.load_model(
            'VGG_19_model.h5',
            custom_objects={'ExponentialDecay': tf.keras.optimizers.schedules.ExponentialDecay}
        )
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_model()

def preprocess_image(image_bytes):
    """Preprocess the image to the format that the model expects."""
    img = Image.open(io.BytesIO(image_bytes)).resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = tf.keras.applications.vgg19.preprocess_input(img_array)  # Preprocess as per VGG19 requirements
    return img_array

@app.route('/')
def index():
    return "Flask server is running."

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        try:
            image_bytes = file.read()
            img_array = preprocess_image(image_bytes)
            
            # Predict using the model
            if model:
                predictions = model.predict(img_array)
                output = np.argmax(predictions, axis=1).tolist()[0]
                
                # Map the prediction to specific rotor types or items
                result_map = {
                    0: "🌀 3 Long Blades Rotor",
                    1: "🌀 3 Short Blade Rotor",
                    2: "🦜 Bird",
                    3: "🦜 Bird + 2 Blade Rotor",
                    4: "✈️ RC Plane",
                    5: "🚁 Drone"
                }
                result = result_map.get(output, "🔍 Unknown Prediction")
                
                # Convert image to base64
                img = Image.open(io.BytesIO(image_bytes))
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                
                return jsonify({
                    'result': result,
                    'img_base64': img_base64
                })
            else:
                return jsonify({'error': 'Model not loaded'}), 500
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
