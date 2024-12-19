import cv2
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
import os
from fer import FER

# Path to save model
model_path = 'data/knn_model.pkl'

# Load the data X_train
with open('data/faces.pkl', 'rb') as w:
    faces = pickle.load(w)

# Load the data y_train
with open('data/names.pkl', 'rb') as f:
    labels = pickle.load(f)

facec = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Jedidi Initialize emotion detector from FER
emotion_detector = FER()

# Vérifiez si le modèle KNN existe déjà
if os.path.exists(model_path):
    print("Chargement du modèle KNN existant...")
    with open(model_path, 'rb') as f:
        knn = pickle.load(f)
else:
    # shape of faces data/matrix
    print('Shape of Faces matrix --> ', faces.shape)
    print("Entraînement d'un nouveau modèle KNN...")

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces,labels)
    
    # Sauvegarder le modèle pour une utilisation future
    with open(model_path, 'wb') as f:
        pickle.dump(knn, f)
    print("Modèle KNN sauvegardé.")

# Define KNN functions


# def distance(x1, x2):
#     d = np.sqrt(((x1 - x2) ** 2).sum())
#     return d


# def knn(xt, X_train=faces, y_train=labels, k=5):
#     vals = []

#     for ix in range(len(labels)):
#         d = distance(X_train[ix], xt)
#         vals.append([d, y_train[ix]])

#     sorted_labels = sorted(vals, key=lambda z: z[0])
#     neighbours = np.asarray(sorted_labels)[:k, -1]

#     freq = np.unique(neighbours, return_counts=True)

#     # freq[0] is list of names and freq[1] is list of counts
#     return freq[0][freq[1].argmax()]

# 0 for default camera
cam = cv2.VideoCapture(0)

#1 for external camera
# cam = cv2.VideoCapture(1)

while True:
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
        face_coordinates = facec.detectMultiScale(gray, 1.3, 5)

         # Draw a rectangle around the face using the coordinates returned by detectMultiScale
        # The rectangle is drawn on the original image (frame)
        for (x, y, w, h) in face_coordinates:
            fc = frame[y:y + h, x:x + w, :]
            r = cv2.resize(fc, (50, 50)).flatten().reshape(1,-1)
            text = knn.predict(r)

            # Jedidi Detect emotions in the current frame
            emotions = emotion_detector.detect_emotions(frame)

            if emotions:
                # Jedidi Get the first detected face's emotions
                emotion_data = emotions[0]['emotions']
                max_emotion = max(emotion_data, key=emotion_data.get)
                (ex, ey, ew, eh) = emotions[0]["box"]


                #cv2.putText(frame, text[0], (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                # Jedidi Display emotion and name on the frame
                cv2.putText(frame, f"{text[0]} - {max_emotion}", (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                #drawing a rectangle around the face for showing
                # represents the top left corner of rectangle
                start_point = (x, y)

                # represents the bottom right corner of rectangle
                end_point = (x + w, y + h)

                # Red color in BGR
                color = (0, 0, 255)

                # Line thickness of 2 px
                thickness = 2
                cv2.rectangle(frame, start_point, end_point, (0, 0, 255), thickness)

        cv2.imshow('Projet face recogonition Ahmed Jedidi 2eme Ingenieurie', frame)
        #ESC key to stop
        if cv2.waitKey(1) == 27:
            break
    else:
        print("error de lecture de la caméra.")
        break

cv2.destroyAllWindows()
