import cv2
import os
import numpy as np
import io

from flask import Flask, flash, request, redirect, url_for, current_app, render_template, send_file
from werkzeug.utils import secure_filename
from multiple_xy_calibration.get_offsets import calc_offsets
from PIL import Image

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

COLORS_NAMES = [
    "Red",
    "Green",
    "Light Blue",
    "Dark Blue",
    "Purple",
    "Black",
    "White",
    "Yellow",
]

app = Flask(__name__)
app.secret_key = 'cetbwh'
#app.config['UPLOAD_FOLDER'] = "/static/"


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/offsets', methods=['POST', 'GET'])
def offsets():
    global COLORS_NAMES
    error = None
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        file = file.read()
        nparr = np.frombuffer(file, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        try:
            width = int(request.form['width'])
            offsets, raw_image = calc_offsets(image, width)
            x_offset, y_offset = offsets
            x_offset.insert(0, 0)
            y_offset.insert(0, 0)
            COLORS_NAMES = COLORS_NAMES[:len(x_offset)]
            if os.path.exists('/static/detected.png'):
                os.remove('/static/detected.png')
            cv2.imwrite(os.path.join("static", 'detected.png'), raw_image)
            return render_template(
                "offsets.html", x_offset=x_offset, y_offset=y_offset, color_names=COLORS_NAMES)
        except:
            flash(u'Exception Occured. Is Image in the format specified?', 'error')
            flash(u'Have you specified ref. square width?', 'error')
            return render_template("index.html", error=error)
    return redirect("/")


if __name__ == '__main__':
    app.run()
