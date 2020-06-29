from flask import Flask, request, Response, jsonify
import os, io
import cv2
import jsonpickle
import base64
from PIL import Image

app = Flask(__name__)

# This end point simply returns the image
@app.route("/image_1", methods=["POST", "GET"])
def image_1():
    f = request.files['image']

    # the image needs to be saved and then read and cannot be used directly because
    # it is of class <class 'werkzeug.datastructures.FileStorage'>
    f.save(f.filename)
    test_image = cv2.imread(f.filename)
    _, img_encoded = cv2.imencode('.png', test_image)
    response = img_encoded.tostring()
    os.remove(f.filename)
    return Response(response=response, status=200, mimetype='image/png')

# This end point returns image as JSON
@app.route("/image_json", methods=["POST", "GET"])
def image_json():
    f = request.files['image']
    f.save(f.filename)
    test_image = cv2.imread(f.filename)

    # ---------------------------------
    # Do some processing
    # ---------------------------------

    test_image = Image.fromarray(test_image.astype("uint8"))
    rawBytes = io.BytesIO()
    test_image.save(rawBytes, "JPEG")
    rawBytes.seek(0)
    img_base64 = base64.b64encode(rawBytes.read())
    return jsonify({'image':str(img_base64), 'image_characteristics':"xxx"})

@app.route("/text", methods=["POST", "GET"])
def text_json():

    # This will be replace by incoming json file
    f = request.files['text']
    f.save(f.filename)

    # Extract the contents of JSON and do some processing

    # ---------------------------------
    # Do some processing
    # ---------------------------------

    some_dictionary = {"prediction_1": 123, "prediction_2": 326, "prediction_3": 456}

    return jsonify(some_dictionary)

if __name__ == "__main__":
    app.run(debug=True)
