import cv2
import numpy as np
import os
import pickle

face_data = []
i = 0

# 0 for default camera
cam = cv2.VideoCapture(0)

#1 for external camera
# cam = cv2.VideoCapture(1)

facec = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

name = input('Ajouter votre nom = ')

ret = True

while(ret):
    # Capture frame-by-frame (ret = True if frame is available)
    # Read the current frame from the webcam
    ret, frame = cam.read()
    if ret == True:
        # Our operations on the frame come here
        # Convert the frame to grayscale
        # Convert the image from BGR to grayscale because Haar Cascades detect faces in grayscale images efficiently
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the current frame
        # scaleFactor: Parameter specifying how much the image size is reduced at each image scale.
        # minNeighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it.
        # minSize: Minimum possible object size. Objects smaller than that are ignored.
        # maxSize: Maximum possible object size. Objects larger than that are ignored.
        # Returns a list of rectangles where each rectangle contains the detected object
        # The rectangles are returned as a list of 4-tuples (x, y, w, h)
        # x, y are the coordinates of the top-left corner of the rectangle
        # w, h are the width and height of the rectangle
        # The detected faces are returned as a list of rectangles
        face_coordinates = facec.detectMultiScale(gray, 1.3, 4)

        # Draw a rectangle around the face using the coordinates returned by detectMultiScale
        # The rectangle is drawn on the original image (frame)
        for (x, y, w, h) in face_coordinates:
            faces = frame[y:y+h, x:x+w, :]
            # faces = cv2.cvtColor(faces, cv2.COLOR_BGR2GRAY)
            # faces = cv2.equalizeHist(faces)
            # faces = cv2.resize(faces, (50, 50))
            # faces = faces.flatten()
            # faces = faces.reshape(1, -1)
            # faces = faces.astype('float32')
            # faces = faces/255.0
            # faces = faces.reshape(1, 50, 50, 3)
            # extract the face from the frame & resize it to 50X50
            resized_faces = cv2.resize(faces, (50, 50))

            # Display the resulting frame after 10 frames
            if i % 10 == 0 and len(face_data) < 10:
                # cv2.imwrite('data/face_' + str(i) + '.jpg', faces)
                # cv2.imwrite('data/face_' + str(i) + '.jpg', resized_faces)
                face_data.append(resized_faces)

            #drawing a rectangle around the face for showing
            # represents the top left corner of rectangle
            start_point = (x, y)

            # represents the bottom right corner of rectangle
            end_point = (x+w, y+h)

            # Blue color in BGR
            color = (255, 0, 0)

            # Line thickness of 2 px
            thickness = 2
            cv2.rectangle(frame, start_point , end_point, color, thickness)
        i += 1

        # Display the resulting frame in a new tab title ="..."
        cv2.imshow('Ahmed Jedidi DSI33', frame)

        # wait for 1ms and if the user press the ESC key or get 10 images then break the loop
        if cv2.waitKey(1) == 27 or len(face_data) >= 10:
            break
    # Problem in camera
    else:
        print('error')
        break

# close all windows
cv2.destroyAllWindows()
# release the camera
cam.release()

# save the face data
# face_data a 10 images of 50X50X3
# Lasa9 les données de la face
face_data = np.asarray(face_data) # convert list to array
#
face_data = face_data.reshape(10, -1)
# ==> 10X7500 array face_data of shape 10X7500 (10 rows, each row depicts one image )
# ==> where 10 depicts the no of images and 7500 depicts the flattened image itself (50X50X3) (structure shown below).

# If not exist create a file ‘names.pkl’ which will contain the same name 10 times
if 'names.pkl' not in os.listdir('data/'):
    names = [name]*10
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
#Else case, means we have our ‘names.pkl’, means it is not the first face we are adding, so just load the ‘names.pkl’ add 10 entries of our current face name and save it as ‘names.pkl’.
else:
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)

    names = names + [name]*10
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

# If not exist create a file ‘faces.pkl’ which will contain the face data

if 'faces.pkl' not in os.listdir('data/'):
    with open('data/faces.pkl', 'wb') as w:
        pickle.dump(face_data, w)
else:
    with open('data/faces.pkl', 'rb') as w:
        faces = pickle.load(w)

    faces = np.append(faces, face_data, axis=0)
    with open('data/faces.pkl', 'wb') as w:
        pickle.dump(faces, w)