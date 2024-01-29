import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from flask import Flask, request, render_template , jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

model_path = 'quantized_disease_model.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

print('Model loaded. Check http://127.0.0.1:5000/')

labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

def getResult(image_path):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = load_img(image_path, target_size=(input_details[0]['shape'][1], input_details[0]['shape'][2]))
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)

    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    return predictions

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')




@app.route('/predictApi', methods=["POST"])
def api():
    # Get the image from post request
    try:
        if 'file' not in request.files:
            return "Please try again. The Image doesn't exist"
        image = request.files.get('file')
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(image.filename))
        image.save(file_path)

        predictions = getResult(file_path)
        predicted_label = labels[np.argmax(predictions)]
        return jsonify({'prediction': predicted_label})
    except:
        return jsonify({'Error': 'Error occur'})




@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        predictions = getResult(file_path)
        predicted_label = labels[np.argmax(predictions)]

        return predicted_label

    return None

if __name__ == '__main__':
    app.run(debug=True)
