import pyttsx3
from imutils.video import FPS
import argparse
import os
import imutils
import cv2
import keras
import tensorflow as tf
from keras.applications.vgg16 import VGG16
from scipy import misc  # run "pip install pillow"
from imutils import face_utils
import dlib
import skvideo
import glob
import imageio
import numpy as np
from keras.models import load_model
#from sklearn.metrics import confusion_matrix, classification_report
#import pandas
skvideo.setFFmpegPath("C:\\ffmpeg\\bin")

engine = pyttsx3.init()


def speak(text):
    engine.setProperty("rate", 160)
    engine.say(text)
    engine.runAndWait()


res = "my name"
speak(res)

predictor_path = 'shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

RECTANGLE_LENGTH = 90

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=40,
                help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=-1,
                help="Whether or not frames should be displayed")
args = vars(ap.parse_args())

print("[INFO] sampling frames from webcam...")
file_lis = glob.glob("./videoss/*.*")
file_lis = file_lis[0]
print(file_lis)


stream = cv2.VideoCapture(file_lis)

fps = FPS().start()
i = 0
# loop over some frames
while fps._numFrames < args["num_frames"]:
    # grab the frame from the stream and resize it to have a maximum
    # width of 400 pixels
    (grabbed, frame) = stream.read()

    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rects = detector(gray, 1)
    rect = rects[0]
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    (x, y, w, h) = face_utils.rect_to_bb(rect)
    w = RECTANGLE_LENGTH
    h = RECTANGLE_LENGTH
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    (x_r, y_r, w_r, h_r) = (x, y, w, h)

    # show the face number
    cv2.putText(frame, "Face #{}".format(0 + 1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    crop_img = frame[y_r:y_r + h_r, x_r:x_r + w_r]

    img_path = 'img'+str(i)+'.jpg'
    i += 1
    img_gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    cv2.imwrite('camera/' + img_path, img_gray)

    # check to see if the frame should be displayed to our screen
    # if args["display"] > 0:
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
stream.release()
cv2.destroyAllWindows()
sequence = []
for file in glob.glob("camera/*.jpg"):
    basename = os.path.basename(file)
    image = imageio.imread(file)
    print(image.shape)
    #image = np.reshape(image, 90 * 90 * 3)
    sequence.append(image)

samples = np.shape(sequence)[0]
h_size = 90
w_size = 90
chanel = 3
np.shape(sequence)
bottleneck_features = np.empty([samples, 2048])
model = VGG16(weights='imagenet', include_top=False)

for j in range(len(sequence)):
    print("Image Number: ", j)
    img = np.expand_dims(sequence[j], axis=0)
    feature = model.predict(img)
    bottleneck_features[j] = feature.flatten()

np.save('camtest', bottleneck_features)
camera_test = np.load('camtest.npy')
model2 = load_model('model.h5')

#print("cameratest shape", camera_test.ndim)
#camera_test = np.expand_dims(camera_test, axis=0)
#camera_test = np.expand_dims(camera_test, axis=0)
#camera_test = np.expand_dims(camera_test, axis=0)
#camera_test = np.expand_dims(Sample_test, axis=0)
#Sample_test = np.expand_dims(Sample_test, axis=0)
#Sample_test = np.expand_dims(Sample_test, axis=0)


#print(" New cameratest dimension", camera_test.ndim)

#camera_test = np.reshape(20*90*90*3)


camera_test1 = np.resize(camera_test, (1268, 20, 90, 90, 3))


prediction = model2.predict(camera_test1)
prediction_result = np.argmax(prediction, axis=1)
class_names = {0: "Stop navigation", 1: "Excuse me", 2: "I am sorry", 3: "Thank you", 4: "Good bye",
               5: "I love this game", 6: "Nice to meet you", 7: "You are Welcome", 8: "How are you?", 9: "Have a good time"}
test_list = []
for s in prediction_result:
    test_list.append(s)
    print(class_names[s])

# print("###############################################################")
res = max(set(test_list), key=test_list.count)
print("Final word")
print(class_names[res])

#Res = class_names[res]
#engine = pyttsx3.init()


# def speak(text):
#     engine.setProperty("rate", 160)
#     engine.say(text)
#     engine.runAndWait()


# #res = "my name"
# speak(Res)
