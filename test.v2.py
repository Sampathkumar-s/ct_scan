# =====================================================
# UNIVERSAL CT SCAN TESTING
# SUPPORTS:
# .nii .png .jpg .jpeg .webp
# =====================================================

!pip install nibabel

# =====================================================
# IMPORT LIBRARIES
# =====================================================

import os
import cv2
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from google.colab import drive
from google.colab import files

from tensorflow.keras.models import load_model

# =====================================================
# MOUNT DRIVE
# =====================================================

drive.mount('/content/drive')

# =====================================================
# LOAD MODEL
# =====================================================

model = load_model(
    "/content/drive/MyDrive/Machine_Learning/ct_scan_model.h5"
)

print("MODEL LOADED")

# =====================================================
# UPLOAD FILE
# =====================================================

uploaded = files.upload()

file_name = list(uploaded.keys())[0]

print("FILE:", file_name)

# =====================================================
# IMAGE SIZE
# =====================================================

IMG_SIZE = 128

# =====================================================
# CHECK FILE EXTENSION
# =====================================================

extension = file_name.split(".")[-1].lower()

# =====================================================
# LOAD .nii FILE
# =====================================================

if extension == "nii":

    nii = nib.load(file_name)

    scan = nii.get_fdata()

    middle_slice = scan[:, :, scan.shape[2] // 2]

    middle_slice = middle_slice - np.min(middle_slice)

    middle_slice = middle_slice / np.max(middle_slice)

    middle_slice = (middle_slice * 255).astype(np.uint8)

    image_data = cv2.resize(
        middle_slice,
        (IMG_SIZE, IMG_SIZE)
    )

    image_data = cv2.cvtColor(
        image_data,
        cv2.COLOR_GRAY2RGB
    )

# =====================================================
# LOAD NORMAL IMAGE
# =====================================================

elif extension in ["png", "jpg", "jpeg", "webp"]:

    img = Image.open(file_name).convert("RGB")

    img = img.resize((IMG_SIZE, IMG_SIZE))

    image_data = np.array(img)

# =====================================================
# INVALID FILE
# =====================================================

else:

    raise ValueError("Unsupported file format")

# =====================================================
# SHOW IMAGE
# =====================================================

plt.imshow(image_data)

plt.title("Uploaded Scan")

plt.axis("off")

plt.show()

# =====================================================
# PREPARE IMAGE
# =====================================================

input_image = image_data / 255.0

input_image = np.expand_dims(input_image, axis=0)

# =====================================================
# PREDICT
# =====================================================

prediction = model.predict(input_image)

score = float(prediction[0][0]) * 100

print("\nCONFIDENCE:", round(score, 2), "%")

# =====================================================
# DISEASE LEVEL
# =====================================================

if score < 33:

    level = "Beginner"

    message = "Low disease level detected"

elif score < 66:

    level = "Intermediate"

    message = "Moderate disease level detected"

else:

    level = "Advanced"

    message = "High disease level detected"

# =====================================================
# FINAL RESULT
# =====================================================

print("\n==========================")

print("DISEASE LEVEL:", level)

print("MESSAGE:", message)

print("CONFIDENCE:", round(score, 2), "%")

print("==========================")