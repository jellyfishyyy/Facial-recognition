import csv
import os

import numpy as np
import tensorflow as tf
from PIL import Image

# from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load the saved model
model_path = "app/model/V2_20_Gender_Age_Prediction"
model = tf.saved_model.load(model_path)


def genderAge(image):
    # Function to preprocess the image
    def preprocess_image(image):
        img = Image.open(os.path.join("app/static/cutting_img/", image)).convert(
            "L"
        )  # Convert to grayscale
        img = img.resize(
            (128, 128), Image.LANCZOS
        )  # Use LANCZOS for high-quality downsampling
        img = np.array(img)
        img = img.reshape(1, 128, 128, 1)
        img = img / 255.0
        return tf.convert_to_tensor(img, dtype=tf.float32)  # Convert to tf.Tensor

    # Function to predict gender and age
    def predict_gender_age(image):
        img = preprocess_image(image)
        predictions = model(inputs=img)
        gender_prediction = predictions[0][0][
            0
        ]  # Update access to the predicted output
        age_prediction = predictions[1][0][0]  # Update access to the predicted output
        gender = "Male" if gender_prediction < 0.5 else "Female"
        age = int(age_prediction)
        return gender, age

    gender, age = predict_gender_age(image)
    print("Image:", image)
    print("Predicted Gender:", gender)
    print("Predicted Age:", age)
    print("-----------------------")

    # Create a CSV file to store the results
    csv_file = "app/data/predictions.csv"
    column_names = ["Image", "Predicted Gender", "Predicted Age"]

    if not os.path.exists(csv_file):
        with open(csv_file, "a", newline="") as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow(column_names)

    # Write the predictions to the CSV file
    with open(csv_file, "a", newline="") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([image, gender, age])
