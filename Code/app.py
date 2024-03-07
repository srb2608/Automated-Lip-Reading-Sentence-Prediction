#from crypt import methods
import os
import glob
from flask import Flask, render_template, request, redirect, url_for
from subprocess import call
import time


app = Flask(__name__)


@app.route("/")
def hello():
    return render_template('main2.html')


@app.route("/main3")
def next():
    return render_template('main3.html')


@app.route('/main3', methods=['GET'])
def test():
    print("GET REQUEST")
    return render_template('main3.html')


@app.route('/main3', methods=['POST'])
def get_data():

    del_files = glob.glob('./camera/*.*')
    for f in del_files:
        os.remove(f)

    del_files1 = glob.glob('./videoss/*.*')
    for f in del_files1:
        os.remove(f)

    videofile = request.files['videofil']
    print(" Post request")

    vid_path = "./videoss/" + videofile.filename
    videofile.save(vid_path)

    # os.remove('result.txt')

    call('python main1.py', shell=True)

    # time.sleep(600)
    #f = open("result.txt", "r")
    #pred = f.read()

    # print(f.read())

    return render_template('main3.html')


app.run(debug=True)
