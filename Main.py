from YOLO import yolo_detection
from Model import Model
from util import utility, end_function
from detection import detection
from xml.dom import minidom
import xml.etree.ElementTree as xml
import os
import cv2
from zipfile import ZipFile
import flask
from flask import Flask, render_template, request, send_file, url_for
from flask_uploads import UploadSet, configure_uploads, IMAGES
from werkzeug import secure_filename
from werkzeug.utils import redirect
from flask.helpers import url_for
import atexit
import random

app = flask.Flask(__name__)
app.config["DEBUG"] = True

def init(file):
    name = random.randint(0, 100)
    directory_main = "Dataset" + str(name)
    parent_dir = "./"
    path_dataset = os.path.join(parent_dir, directory_main)
    os.mkdir(path_dataset)
    directory_img = "images"
    directory_xml = "XML Files"
    parent_dir = path_dataset
    path_img = os.path.join(parent_dir, directory_img)
    path_xml = os.path.join(parent_dir, directory_xml)
    # os.rmdir(path_img)
    os.mkdir(path_img)
    # os.rmdir(path_xml)
    os.mkdir(path_xml)
    # Opens the Video file
    cap = cv2.VideoCapture(file)
    i = 0
    while (cap.isOpened()):
        print (i)
        ret, frame = cap.read()
        if (cv2.waitKey(25) & 0xFF == ord('q')) or ret == False:
            break
        #cv2.imshow('frame', frame)
        cv2.imwrite(os.path.join(path_img, str(i) + '.jpg'), frame)
        #img_path = 'frame.jpg'
        # LOAD IMAGE
        detection(frame, i, path_xml)
        i = i+1
        if i == 2:
            break


    cap.release()
    cv2.destroyAllWindows()

    with ZipFile('Dataset.zip', 'w') as zipObj:
        # Iterate over all the files in directory
        for folderName, subfolders, filenames in os.walk(directory_main):
            for filename in filenames:
                # create complete filepath of file in directory
                filePath = os.path.join(folderName, filename)
                # Add file to zip
                zipObj.write(filePath)





@app.route('/uploader', methods = ['GET', 'POST'])
def upload_file():
    directory_main = "Dataset.zip"
    parent_dir = "./"
    path_dataset = os.path.join(parent_dir, directory_main)
    if request.method == 'POST':
        f = request.files['file']
        print(f.filename)
        f.save(secure_filename(f.filename))
        init(f.filename)
        end_function()
        try:
            return send_file(path_dataset,
                           attachment_filename='Dataset.zip')
        except Exception as e:
            return str(e)
#...................

@app.route('/')
def login():
    error = None
    return render_template('login.html', error=error)




@app.route('/aboutUs')
def aboutus():
    return render_template('aboutUs.html')


@app.route('/signin', methods=['GET', 'POST'])
def sign_in():
    if request.method == 'POST':
        if request.method == 'POST':
            if True:
                print(request.form['username'], request.form['password'])
                return render_template('index.html')
                # error = 'Invalid Credentials. Please try again.'
            else:
                return redirect(url_for('home'))



#...................


if __name__ == '__main__':
    app.run(debug = True)