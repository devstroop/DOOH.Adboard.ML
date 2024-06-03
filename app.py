#!/usr/bin/env python

import cv2
from flask import Flask, render_template, Response
import base64

app = Flask(__name__)
vc = cv2.VideoCapture(-1)
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

detected_faces = []  # List to store detected faces

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def gen_frames():
    global detected_faces
    while True:
        ret, frame = vc.read()
        if not ret:
            break
        
        # Detect faces
        detected_faces = detect_faces(frame)
        
        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html', detected_faces=detected_faces)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
