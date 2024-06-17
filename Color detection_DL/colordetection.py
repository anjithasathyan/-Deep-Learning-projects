import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# color detection

img = cv.imread("traffic.jpg")

def detectRed(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # red color range
    lower_range = (0,100,100)
    upper_range = (10,255,255)

    mask = cv.inRange(hsv, lower_range, upper_range)
    masked_image = cv.bitwise_and(img, img, mask=mask)
    mask1 = cv.medianBlur(mask, 9)
    contours, _ = cv.findContours(mask1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = contours[0]
        x,y,w,h = cv.boundingRect(cnt)
        cv.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 3)
        cv.putText(img, "RED", (x,y), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,255), 2)
    else:
        print("color not detected")

def detectYellow(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # yellow color range
    lower_range = (25,100,100)
    upper_range = (35,255,255)

    mask = cv.inRange(hsv, lower_range, upper_range)
    masked_image = cv.bitwise_and(img, img, mask=mask)
    mask1 = cv.medianBlur(mask, 9)
    contours, _ = cv.findContours(mask1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = contours[0]
        x,y,w,h = cv.boundingRect(cnt)
        cv.rectangle(img, (x,y), (x+w, y+h), (0,255,255), 3)
        cv.putText(img, "Yellow", (x,y), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,255), 2)
    else:
        print("color not detected")

def detectGreen(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # green color range
    lower_range = (45,100,100)
    upper_range = (65,255,255)

    mask = cv.inRange(hsv, lower_range, upper_range)
    masked_image = cv.bitwise_and(img, img, mask=mask)
    mask1 = cv.medianBlur(mask, 9)
    contours, _ = cv.findContours(mask1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = contours[0]
        x,y,w,h = cv.boundingRect(cnt)
        cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)
        cv.putText(img, "green", (x,y), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,0), 2)
    else:
        print("color not detected")


detectRed(img)
detectYellow(img)
detectGreen(img)
cv.imshow('Original', img)
#cv.imshow('Original1', mask1)


cv.waitKey(0)
cv.destroyAllWindows()