#!/usr/bin/env python

# Import libraries
import cv2
import numpy as np
import dlib
from flask import Flask, render_template, Response
import io

# Initialize the Flask app
app = Flask(__name__)

# Initialize the video capture object
vc = cv2.VideoCapture(0)

# Initialize the face detector
detector = dlib.get_frontal_face_detector()

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def gen():
    """Video streaming generator function."""
    while True:
        # Capture frame-by-frame
        read_return_code, frame = vc.read()
        if not read_return_code:
            continue
        
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = detector(gray)
        
        # Draw rectangles around detected faces
        for i, face in enumerate(faces):
            x, y = face.left(), face.top()
            x1, y1 = face.right(), face.bottom()
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
            cv2.putText(frame, f'Face {i+1}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Encode the frame to JPEG format
        encode_return_code, image_buffer = cv2.imencode('.jpg', frame)
        
        # Convert the encoded image to bytes
        io_buf = io.BytesIO(image_buffer)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + io_buf.read() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, threaded=True)
