import cv2
import sys
import logging as log
import datetime as dt
from time import sleep

eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')
left_ear_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_mcs_leftear.xml')
right_ear_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_mcs_rightear.xml')

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0


if eye_cascade.empty():
  raise IOError('Unable to load the eye cascade classifier xml file')

if left_ear_cascade.empty():
  raise IOError('Unable to load the left ear cascade classifier xml file')

if right_ear_cascade.empty():
  raise IOError('Unable to load the right ear cascade classifier xml file')

if face_cascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file')

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

     # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        left_ear = left_ear_cascade.detectMultiScale(roi_gray, 1.3, 5)
        right_ear = right_ear_cascade.detectMultiScale(roi_gray, 1.3, 5)

        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey),(ex+ew,ey+eh),(0, 0, 255), 3)
            #pixellate(ew, eh, 4, roi_color)
    
        for (x,y,w,h) in left_ear:
            cv2.rectangle(roi_color, (x,y), (x+w,y+h), (255, 0, 0), 3)
            #pixellate(ew, eh, 4, roi_color)

        for (x,y,w,h) in right_ear:
            cv2.rectangle(roi_color, (x,y), (x+w,y+h), (255, 0, 0), 3)
            #pixellate(ew, eh, 4, roi_color)

    

    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

def pixellate(width, height, pxSize, color):
    for x in range(0, width, pxSize):
        for y in range(0, height, pxSize):
            cv2.rectangle(color, (x,y), (x+pxSize,y+pxSize), (255, 0, 0), 3)


# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
