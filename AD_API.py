# Dependencies
from flask import Flask, jsonify, request
from tensorflow import keras
import numpy as np
import cv2
import zipfile

with zipfile.ZipFile('model_1.zip', 'r') as zip_file:
    zip_file.extractall()

AD_API = Flask(__name__)

# Load the saved tf model from the saved_model directory

saved_model_dir = 'model_1'

model = keras.models.load_model(saved_model_dir)

# class labels

labels = ['AD', 'CN', 'EMCI', 'MCI']


@AD_API.route('/classify_image_ad', methods=['POST'])
def classify_image_ad():
    # get an image from the request (key=image)

    image_file = request.files['image']

    # load the image and preprocess it
    input_image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    input_image = cv2.resize(input_image, (128, 128))  # Reshape to 128x128
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32') / 255.0  # Normalize

    # Classify the received image using the loaded model
    prediction = model.predict(input_image)

    # Predicted class label
    predicted_class_index = np.argmax(prediction[0])
    predicted_class_label = labels[predicted_class_index]

    # Returning the predicted class as a JSON response
    return jsonify({'The MRI image belongs to the class': predicted_class_label})


if __name__ == '__main__':
    ''' Run the app'''
    AD_API.run()


