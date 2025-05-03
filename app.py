from flask import Flask, render_template, request
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)

# Load the trained model
model = load_model('model/roman_numeral_predictor.keras')

# Load the label encoder
label_encoder = joblib.load('model/label_encoder.pkl')  # You need to save this from your training code

# Ensure upload folder exists
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', prediction='No image uploaded')

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', prediction='No selected file')

    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Preprocess the image (must match training input: 50x50 grayscale)
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (50, 50))
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        image = np.expand_dims(image, axis=-1) # Add channel dimension
        image = image.astype('float32') / 255.0

        # Predict
        prediction = model.predict(image)
        predicted_index = np.argmax(prediction)
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]

        return render_template('index.html', prediction=predicted_label, image_path=filepath)

    return render_template('index.html', prediction='Error during prediction')

if __name__ == '__main__':
    app.run(debug=True)
