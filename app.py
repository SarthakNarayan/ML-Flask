from flask import Flask, render_template, redirect, request, Response, send_file
import numpy as np
import os
import matplotlib.image as mpimg
from predictions.predictor import Segmentation
import matplotlib.pyplot as plt
import cv2


app = Flask(__name__)

# For renaming the incoming input image
name_file = None


@app.route("/", methods=["POST", "GET"])
def home_page():
    global name_file

    if name_file is not None:
        if os.path.exists(name_file):
            os.remove(name_file)
            name_file = None

    if request.method == "GET":
        return render_template("index.html")
    else:
        f = request.files['image']
        f.save(os.path.join("static/output_images", f.filename))
        name_file = os.path.join("static/output_images", f.filename)
        return redirect("/submit")


@app.route("/submit", methods=["POST", "GET"])
def submit():
    global name_file
    segmentor = Segmentation(name_file)
    mask = segmentor.segment()
    mpimg.imsave(os.path.join("static/output_images", "out.png"), mask)
    return redirect("/output")


@app.route("/output")
def output():
    global name_file
    return render_template("output.html", input_image=name_file, output_image=os.path.join("static/output_images", "out.png"))

"In this end point image is directly returned"
@app.route("/return_1")
def return_1():
    pass

if __name__ == "__main__":
    app.run(debug=True)
