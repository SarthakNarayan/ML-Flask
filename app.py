from flask import Flask, render_template, redirect, request
import numpy as np
import os
import matplotlib.image as mpimg
from predictions.predictor import Segmentation
import matplotlib.pyplot as plt

app = Flask(__name__)

# For renaming the incoming input image
name_file = None
@app.route("/", methods=["POST", "GET"])
def home_page():
    global name_file
    if request.method == "GET":
        return render_template("index.html")
    else:
        f = request.files['image']
        f.save(f.filename)
        name_file = f.filename
        return redirect("/submit")

@app.route("/submit", methods=["POST", "GET"])
def submit():
    global name_file
    segmentor = Segmentation(name_file)
    mask = segmentor.segment()
    mpimg.imsave("static/output_images/out.png", mask)
    os.remove(name_file)
    return redirect("/output")

@app.route("/output")
def output():
    return render_template("output.html" , output_image = "static/output_images/out.png")

if __name__ == "__main__":
    app.run(debug=True)