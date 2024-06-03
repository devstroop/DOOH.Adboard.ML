# Import libraries 
import cv2 
import numpy as np 
import dlib 


if(__name__ == '__main__'):
  while True: 
    # Capture frame-by-frame 
    ret, frame = cap.read() 
    frame = cv2.flip(frame, 1) 

    # Our operations on the frame come here 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    faces = detector(gray) 

    # Counter to count number of faces 
    i = 0
    for face in faces: 
      x, y = face.left(), face.top() 
      x1, y1 = face.right(), face.bottom() 
      cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2) 

      # Increment the iterartor each time you get the coordinates 
      i = i+1

      # Adding face number to the box detecting faces 
      cv2.putText(frame, 'face num'+str(i), (x-10, y-10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) 
      print(face, i) 

    # Display the resulting frame 
    cv2.imshow('frame', frame) 

