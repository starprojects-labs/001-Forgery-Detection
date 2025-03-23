import io
import cv2
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# ml model -> https://drive.google.com/file/d/1X5o0cqvjTHoWLqfDyw-_E7Z5wpXqyCoS/view?usp=drive_link
model = load_model("sucessForgery.keras")

print("Model loaded successfully!")

@app.route('/', methods=['GET'])
def home():
    return jsonify("Hello darling")

@app.route('/uploadfile', methods=['POST'])
def upload_data():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    try:
        image = Image.open(io.BytesIO(file.read()))
        image = image.convert('RGB')
        img = cv2.imread(image)
        if img is not None:
            img = cv2.resize(img, (224, 224))
            img = img / 255.0
            img = np.expand_dims(img, axis=0)
            print("Image preprocessed successfully!")
            modelRunner(img)
            result = True
            return jsonify({"result": result}), 200
        else:
            return jsonify({"error": "Failed to process image"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def modelRunner(img):
    predictions = model.predict(img)
    print(predictions)
    predicted_class = 1 if predictions[0][0] > 0.5 else 0

    if predicted_class == 0:
        print("Prediction: Authentic Image ✅")
    else:
        print("Prediction: Tampered Image ❌")

if __name__ == '__main__':
    app.run(debug=True, port=5000)
