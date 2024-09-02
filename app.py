from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img 
import numpy as np

from tensorflow.keras.optimizers.schedules import ExponentialDecay

# Load the VGG-19 model from the .h5 file with custom objects
model = tf.keras.models.load_model(
    'VGG_19_model.h5',
    custom_objects={'ExponentialDecay': ExponentialDecay}
)

# Initialize the Flask application
app = Flask(__name__)

def preprocess_image(image_path):
    """Preprocess the image to the format that the model expects."""
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = tf.keras.applications.vgg19.preprocess_input(img_array)  # Preprocess as per VGG19 requirements
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        file_path = './' + file.filename
        file.save(file_path)

        # Preprocess the image
        img_array = preprocess_image(file_path)
        
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
        elif output == 4:
            result = "drone"
        else:
            result = "Unknown prediction"

        return jsonify({'prediction': output, 'result': result})

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to ensure the server is running."""
    return jsonify({'status': 'Healthy'}), 200

if __name__ == '__main__':
    app.run(debug=True)
