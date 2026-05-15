# =====================================================
# CT SCAN MODEL TRAINING
# =====================================================

!pip install nibabel

# =====================================================
# IMPORT LIBRARIES
# =====================================================

import os
import cv2
import json
import zipfile
import random
import nibabel as nib
import numpy as np

from google.colab import drive

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras import layers, models

# =====================================================
# MOUNT GOOGLE DRIVE
# =====================================================

drive.mount('/content/drive')

# =====================================================
# ZIP FILE PATH
# =====================================================

zip_path = "/content/drive/MyDrive/Machine_Learning/ct_scan.zip"

# =====================================================
# EXTRACT DATASET
# =====================================================

extract_path = "/content/ct_scan"

os.makedirs(extract_path, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:

    zip_ref.extractall(extract_path)

print("DATASET EXTRACTED")

# =====================================================
# DATASET PATH
# =====================================================

base_path = "/content/ct_scan/Abdominal CT scans/Abdominal CT scans/images"

# =====================================================
# GET NII FILES
# =====================================================

nii_files = []

folders = os.listdir(base_path)

for folder in folders:

    folder_path = os.path.join(base_path, folder)

    if os.path.isdir(folder_path):

        files_inside = os.listdir(folder_path)

        for file in files_inside:

            if file.endswith(".nii"):

                full_path = os.path.join(folder_path, file)

                nii_files.append(full_path)

print("TOTAL FILES:", len(nii_files))

# =====================================================
# LOAD CT SCANS
# =====================================================

images = []

labels = []

IMG_SIZE = 128

for file_path in nii_files:

    try:

        nii = nib.load(file_path)

        scan = nii.get_fdata()

        middle_slice = scan[:, :, scan.shape[2] // 2]

        middle_slice = middle_slice - np.min(middle_slice)

        middle_slice = middle_slice / np.max(middle_slice)

        middle_slice = (middle_slice * 255).astype(np.uint8)

        middle_slice = cv2.resize(
            middle_slice,
            (IMG_SIZE, IMG_SIZE)
        )

        middle_slice = cv2.cvtColor(
            middle_slice,
            cv2.COLOR_GRAY2RGB
        )

        images.append(middle_slice)

        # TEMP LABELS
        labels.append(random.randint(0,1))

    except Exception as e:

        print("ERROR:", e)

# =====================================================
# NUMPY
# =====================================================

X = np.array(images) / 255.0

y = np.array(labels)

# =====================================================
# SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.2,

    random_state=42

)

# =====================================================
# MODEL
# =====================================================

model = models.Sequential([

    layers.Input(shape=(128,128,3)),

    layers.Conv2D(32, (3,3), activation='relu'),

    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),

    layers.MaxPooling2D(),

    layers.Conv2D(128, (3,3), activation='relu'),

    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(128, activation='relu'),

    layers.Dropout(0.5),

    layers.Dense(1, activation='sigmoid')

])

# =====================================================
# COMPILE
# =====================================================

model.compile(

    optimizer='adam',

    loss='binary_crossentropy',

    metrics=['accuracy']

)

# =====================================================
# TRAIN
# =====================================================

model.fit(

    X_train,
    y_train,

    validation_data=(X_test, y_test),

    epochs=10,

    batch_size=16

)

# =====================================================
# SAVE MODEL
# =====================================================

save_path = "/content/drive/MyDrive/Machine_Learning"

model.save(f"{save_path}/ct_scan_model.h5")

print("MODEL SAVED")