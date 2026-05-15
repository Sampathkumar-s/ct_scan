from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# ======================================
# LOAD MODEL
# ======================================

model = load_model('ct_scan_model.h5')

# ======================================
# FLASK APP
# ======================================

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ======================================
# IMAGE SIZE
# ======================================

IMG_SIZE = 128

# ======================================
# HOME PAGE
# ======================================

@app.route('/')
def home():
    return render_template('index.html')

# ======================================
# PREDICTION
# ======================================

@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return 'No file uploaded'

    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

    file.save(file_path)

    # ==============================
    # LOAD IMAGE
    # ==============================

    img = image.load_img(
        file_path,
        target_size=(IMG_SIZE, IMG_SIZE)
    )

    img_array = image.img_to_array(img)

    img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    # ==============================
    # PREDICT
    # ==============================

    prediction = model.predict(img_array)

    score = float(prediction[0][0])

    # ==============================
    # DISEASE LEVEL
    # ==============================

    if score < 0.33:
        disease_level = 'Beginner'
        message = 'Low level disease detected'

    elif score < 0.66:
        disease_level = 'Intermediate'
        message = 'Moderate level disease detected'

    else:
        disease_level = 'Advanced'
        message = 'High level disease detected'

    confidence = round(score * 100, 2)

    return render_template(
        'index.html',
        prediction=disease_level,
        confidence=confidence,
        message=message,
        image_path=file_path
    )

# ======================================
# RUN APP
# ======================================

if __name__ == '__main__':
    app.run(debug=True)