import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
name = input("what is your name?")
cap = cv2.VideoCapture(0)
skip = 0
face_data = []
face_cas = cv2.CascadeClassifier(".../haarcascade_frontalface_default.xml")
while True:

    ret, frame = cap.read()
    gr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if (ret == False):
        continue
    faces = face_cas.detectMultiScale(frame, 1.3, 5)
    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)

    for face in faces:
        x, y, w, h = face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
        snip = frame[y - 10:y + h + 10, x - 10:x + w + 10]
        snip = cv2.resize(snip, (100, 100))
        skip += 1
        cv2.imshow("snip", snip)
        if (skip % 10 == 0):
            face_data.append(snip)
            print(len(face_data))
    cv2.imshow("frame", frame)

    pressed = cv2.waitKey(5) & 0xFF
    if pressed == ord("q") or skip == 100:
        break
face_data = np.asarray(face_data)
face_data = face_data.reshape(face_data.shape[0], -1)
print(face_data.shape)
np.save(".../"+name+".npy", face_data)
cap.release()
cv2.destroyAllWindows()
