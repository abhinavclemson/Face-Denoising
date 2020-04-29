import cv2
import numpy as np
import os
import sys

# Using xml Haar casacade for frontal face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# mouth_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

font = cv2.FONT_HERSHEY_SIMPLEX


# Import video using open cv
cap = cv2.VideoCapture(0)

# width of the frame of the input video
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)

# height of the frame of the input video
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)

size = (width, height)

# for exporting video as .mp4
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# for writing each output frame in asg04.mp4 file as output with 30 frame per second
out = cv2.VideoWriter('asg05.mp4', fourcc, 30.0, size)


def face_object(img, name):
    # Creating a face object for implementing Haar casacades on it

    # converting image frame into grayscale, so it is easier to find face
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # function with scale factor(how much the image size is reduced at each image scale) of 1.1  and minNeighbors will be 5, to draw rectangle on the faces detect
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    i = 0
    for (x, y, w, h) in faces:
        # drawing rectangle on each detected face
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # region of interest in colored frame
        roi_color = img[y:y + h, x:x + w]

        # region of interest in gray frame
        roi_gray = gray[y:y + h, x:x + w]

        cv2.putText(roi_color, 'face_' + str(i), (0, 10), font, 0.5, (255, 255, 255), 2)

        i += 1

        # for detecting eyes in all gray regions of interest
        eyes = eyes_cascade.detectMultiScale(roi_gray)

        # iterating list of all eyes in a given frame
        no_of_eyes = 0
        for (ex, ey, ew, eh) in eyes:
            # drawing rectangle on each eye
            no_of_eyes += 1
            #cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

            if no_of_eyes>=2:
                cv2.imwrite("output/face" + str(name) +'_'+str(i)+".jpg", roi_color)

        # iterating list of all smiles in a given frame

        # uncomment it to see buggy mouth and smile casacades
        '''
        #for detecting smiles in all gray region of interest

        mouths= mouth_cascade.detectMultiScale(roi_gray)
        for (mx, my, mw, mh) in mouths:
            #drawing rectangle on each smile 
            cv2.rectangle(roi_color, (mx,my), (mx+mw, my+mh),(0, 255, 0), 2)
        '''

name=0

while True:
    # reading each frame from given input video
    ret, frame = cap.read()
    # applying face_object method on each frame
    if ret:
        face_object(frame, name)
        name+=1
        out.write(frame)
    else:
        # if there is no frame
        break

    # displaying each output frame

    cv2.imshow('', frame)

    # press 'q' to quit and save your output file
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()