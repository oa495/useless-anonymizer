import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
left_ear_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_mcs_leftear.xml')
right_ear_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_mcs_rightear.xml')
nose_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_mcs_nose.xml')

img = cv2.imread('face.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
h = img.shape[0]
w = img.shape[1]

if eye_cascade.empty():
  raise IOError('Unable to load the eye cascade classifier xml file')

if left_ear_cascade.empty():
  raise IOError('Unable to load the left ear cascade classifier xml file')

if right_ear_cascade.empty():
  raise IOError('Unable to load the right ear cascade classifier xml file')

if nose_cascade.empty():
  raise IOError('Unable to load the right ear cascade classifier xml file')



eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
left_ear = left_ear_cascade.detectMultiScale(gray, 1.3, 5)
right_ear = right_ear_cascade.detectMultiScale(gray, 1.3, 5)
nose = nose_cascade.detectMultiScale(gray, 1.3, 5)

def getBiometric(point):
    for (x, y, w, h) in eyes:
        eye_rect = Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
        cv2.rectangle(img, (x, y), (x + w, y + h),(0, 0, 255), 3)
        if eye_rect.contains(point):
            return True

    for (x, y, w, h) in left_ear:
        left_ear_rect = Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)
        if left_ear_rect.contains(point):
            return True

    for (x, y, w, h) in right_ear:
        right_ear_rect = Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
        cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 0), 3)
        if right_ear_rect.contains(point):
            return True

    for (x, y, w, h) in nose:
        nose_rect = Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
        cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 5)
        if nose_rect.contains(point):
            return True

for y in range(0, h):
    for x in range(0, w):
        if (not (getBiometric(Point(y, x)))):
            img[y, x] = [100, 0, 0]

cv2.imshow('image', img)
cv2.waitKey()
cv2.destroyAllWindows()