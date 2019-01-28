import base64
import numpy as np
import io
from PIL import Image
import keras
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

classes = ["cloud", "moon", "rainbow", "star", "sun"]

def get_model():
    global model
    model = load_model("model_new.h5")
    print("Model Loaded")
    
def preprocess_img(image, target_size, inv):
    image = image.convert("L")
    image = image.resize(target_size)
    if inv==True :
        image=np.invert(image)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image    

print("loading model...")
get_model()

@app.route('/')
def index():
	return render_template("index.html")
    

@app.route("/predict-image/", methods = ["GET","POST"])
def predict_img():
    message = request.get_json(force=True)
    encoded = message["image"]
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_img = preprocess_img(image, target_size=(28,28), inv=False)
    pred = model.predict(processed_img)
    idx = np.argmax(np.array(pred[0]))
    response = {
            'predictionImg' : str(classes[idx])
    }
    return jsonify(response)

@app.route("/predict-drawing/", methods = ["GET","POST"])
def predict_draw():
    message = request.get_json(force=True)
    encoded = message["image"]
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_img = preprocess_img(image, target_size=(28,28), inv=True)
    pred = model.predict(processed_img)
    idx = np.argmax(np.array(pred[0]))
    response = {
            'predictionDraw' : str(classes[idx])
    }
    return jsonify(response)







