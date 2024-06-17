import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Face Detection
face_cascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

img = cv.imread("singleman.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 4)

print(faces)

for x,y,w,h in faces:
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 4)

cv.imshow('original', img)
cv.imshow('gray', gray)

cv.waitKey(0)
cv.destroyAllWindows()