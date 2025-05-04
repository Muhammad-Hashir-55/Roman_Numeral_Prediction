from flask import Flask, render_template, request
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model # type: ignore
from sklearn.preprocessing import LabelEncoder
import joblib
import PIL.Image
import google.generativeai as genai
import api  # Your API key module

app = Flask(__name__)

# Load the CNN model for single-character Roman numeral prediction
model = load_model(r'model/roman_numeral_predictor.keras')
label_encoder = joblib.load('model/label_encoder.pkl')

# Upload folder
UPLOAD_FOLDER = r'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure Gemini API
genai.configure(api_key='AIzaSyAEhUW514BM2cd_4R9O7FGeW3KBSa647ss')
gemini_model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp-1219")

# -------------- Routes --------------

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

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

        # Preprocess for CNN
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (50, 50))
        image = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=-1)
        image = image.astype('float32') / 255.0

        prediction = model.predict(image)
        predicted_index = np.argmax(prediction)
        predicted_label = label_encoder.inverse_transform([predicted_index])[0]

        # Convert predicted Roman numeral to integer
        roman_value = roman_to_int(predicted_label)

        # Format prediction output
        final_prediction = f"{predicted_label} → {roman_value}"

        # Prepare relative path for HTML
        relative_path = os.path.join('static', 'uploads', file.filename)

        return render_template('index.html', prediction=final_prediction, image_path=relative_path)

    return render_template('index.html', prediction='Error during prediction')



@app.route('/gemini')
def gemini_index():
    return render_template('gemini.html', prediction=None)

@app.route('/gemini_predict', methods=['POST'])
def gemini_predict():
    if 'image' not in request.files:
        return render_template('gemini.html', prediction='No image uploaded')

    file = request.files['image']
    if file.filename == '':
        return render_template('gemini.html', prediction='No selected file')

    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Load image using PIL
        img = PIL.Image.open(filepath)

        # Ask Gemini to read the Roman numeral
        response = gemini_model.generate_content(
            [
                "Read and return exactly what is written in this image (assume it's a Roman numeral).",
                img
            ]
        )

        roman_text = response.text.strip()

        # Convert to integer
        roman_value = roman_to_int(roman_text)

        relative_path = os.path.join('static', 'uploads', file.filename)
        return render_template('gemini.html',
                       prediction=f'{roman_text} → {roman_value}',
                       image_path=relative_path)


    return render_template('gemini.html', prediction='Error during prediction')


# -------------- Utility --------------

def roman_to_int(s: str) -> int:
    roman_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    total = 0
    prev = 0
    for char in reversed(s.upper()):
        curr = roman_map.get(char, 0)
        if curr < prev:
            total -= curr
        else:
            total += curr
        prev = curr
    return total

# -------------- Run Server --------------

if __name__ == '__main__':
    app.run(debug=True)
