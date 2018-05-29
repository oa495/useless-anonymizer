import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
import threading
import json

from SimpleWebSocketServer import SimpleWebSocketServer, WebSocket
server = None
clients = []

class SimpleWSServer(WebSocket):
    def handleConnected(self):
        clients.append(self)

    def handleClose(self):
        clients.remove(self)

def run_server():
    global server
    server = SimpleWebSocketServer('', 9000, SimpleWSServer,
                                   selectInterval=(1000.0 / 15) / 1000)
    server.serveforever()

t = threading.Thread(target=run_server)
t.start()

# The rest of the OpenCV code ...

eye_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_eye.xml')
left_ear_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_mcs_leftear.xml')
right_ear_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_mcs_rightear.xml')
nose_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_mcs_nose.xml')
face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_default.xml')

log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)

if eye_cascade.empty():
  raise IOError('Unable to load the eye cascade classifier xml file')

if left_ear_cascade.empty():
  raise IOError('Unable to load the left ear cascade classifier xml file')

if right_ear_cascade.empty():
  raise IOError('Unable to load the right ear cascade classifier xml file')

if nose_cascade.empty():
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
    biometrics = {}

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

     # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        left_ear = left_ear_cascade.detectMultiScale(roi_gray, 1.3, 5)
        right_ear = right_ear_cascade.detectMultiScale(roi_gray, 1.3, 5)
        nose = nose_cascade.detectMultiScale(roi_gray, 1.3, 5)

        for (x,y,w,h) in eyes:
            biometrics['eyes'] = [x, y, w, h]
            cv2.rectangle(roi_color, (x,y),(x+w,y+h),(0, 0, 255), 3)

        for (x,y,w,h) in left_ear:
            biometrics['left_ear'] = [x, y, w, h]
            cv2.rectangle(roi_color, (x,y), (x+w,y+h), (255, 0, 0), 3)

        for (x,y,w,h) in right_ear:
            biometrics['right_ear'] = [x, y, w, h]
            cv2.rectangle(roi_color, (x,y), (x+w,y+h), (255, 0, 0), 3)

        for (x,y,w,h) in nose:
            biometrics['nose'] = [x, y, w, h]
            cv2.rectangle(roi_color, (x,y), (x+w,y+h), (255, 0, 0), 3)

    def dumper(obj):
        try:
            return obj.toJSON()
        except:
            return obj.__dict__

    print json.dumps(biometrics, default=dumper, indent=2)
    # for client in clients:
    #     msg = json.dumps(biometrics, default=set_default)
    #     client.sendMessage(unicode(msg))

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
