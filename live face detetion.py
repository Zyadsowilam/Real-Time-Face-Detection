# Face detection
# get data here as an example is cropped faces
# Make it black and white to decrese the features needed
# Many Angles
# cv2.CascadeClassifier to ake classifier
# cv2.imread to read image
# image is just an array
# cv2.imshow show img, cv2.waitkey to wait
# cvtColor convert color
# detectMultiScale to detect the lasssifier in the img in all scale
# to draw rec use color img  ,upper lef tuble coord, lower+upper right+left coor , color , thick
# VideCapture take video VideCapture(0)take video from default cam ot give it video
# frame is the image in video
# cv2.waitKey(1) for live
import cv2

from random import randrange

# pretrained face
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# read face
# me = cv2.imread('dat.jpg')

# to use webcam
webcam = cv2.VideoCapture('test.mp4')

while True:
    # read the current time
    successful_frame_read, frame = webcam.read()
    # gray Scale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces give coordinates of rectangle surround the face
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    # Draw rec around the face
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(128, 256), randrange(128, 256), randrange(128, 256)), 2)

    cv2.imshow("Face detect", frame)
    key = cv2.waitKey(1)
    # to quit press q
    if key == 81 or key == 113:
        break
