# =====================================================
# CT SCAN TESTING
# =====================================================

!pip install nibabel

# =====================================================
# IMPORT LIBRARIES
# =====================================================

import nibabel as nib
import numpy as np
import cv2
import matplotlib.pyplot as plt

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
# UPLOAD CT SCAN
# =====================================================

uploaded = files.upload()

file_name = list(uploaded.keys())[0]

print("FILE:", file_name)

# =====================================================
# LOAD CT SCAN
# =====================================================

nii = nib.load(file_name)

scan = nii.get_fdata()

middle_slice = scan[:, :, scan.shape[2] // 2]

middle_slice = middle_slice - np.min(middle_slice)

middle_slice = middle_slice / np.max(middle_slice)

middle_slice = (middle_slice * 255).astype(np.uint8)

IMG_SIZE = 128

middle_slice = cv2.resize(
    middle_slice,
    (IMG_SIZE, IMG_SIZE)
)

middle_slice = cv2.cvtColor(
    middle_slice,
    cv2.COLOR_GRAY2RGB
)

# =====================================================
# SHOW IMAGE
# =====================================================

plt.imshow(middle_slice)

plt.title("CT SCAN")

plt.axis("off")

plt.show()

# =====================================================
# PREPARE IMAGE
# =====================================================

input_image = middle_slice / 255.0

input_image = np.expand_dims(input_image, axis=0)

# =====================================================
# PREDICT
# =====================================================

prediction = model.predict(input_image)

score = float(prediction[0][0]) * 100

print("\nCONFIDENCE:", score)

# =====================================================
# DISEASE LEVEL
# =====================================================

if score < 33:

    level = "Beginner"

    message = "Low disease level"

elif score < 66:

    level = "Intermediate"

    message = "Moderate disease level"

else:

    level = "Advanced"

    message = "High disease level"

# =====================================================
# FINAL RESULT
# =====================================================

print("\n========================")

print("DISEASE LEVEL:", level)

print("MESSAGE:", message)

print("CONFIDENCE:", round(score, 2), "%")

print("========================")