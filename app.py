from flask import Flask, request, jsonify, send_from_directory, render_template
import pickle
import numpy as np
import cv2
import base64
from flask_cors import CORS
# run the flask application using python app.py for prediction

app = Flask(__name__, static_folder='static')
CORS(app)
# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Define the labels manually
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Decode image from base64
    image_data = base64.b64decode(data['image'].split(',')[1])
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Example feature extraction (modify this based on your requirements)
    # You need to replace the following line with your actual hand landmark processing logic.
    data_aux = np.random.rand(42).tolist()  # Dummy data for illustration

    prediction = model.predict([np.asarray(data_aux)])  # Replace with actual feature extraction
    predicted_character = labels_dict[int(prediction[0])]

    return jsonify({"prediction": predicted_character})

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080)
