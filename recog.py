import numpy as np
import cv2
import os


def dist(x1, x2):
    return np.sqrt(sum((x1 - x2)**2))


def knn(X, Y, X_test, k=5):
    vals = []
    for i in range(X.shape[0]):
        vals.append((dist(X_test, X[i]), Y[i]))
    vals = sorted(vals)[:k]
    vals = np.array(vals)
    new_vals = np.unique(vals[:, 1], return_counts=True)
    index = new_vals[1].argmax()
    return new_vals[0][index]


cap = cv2.VideoCapture(0)
dataset_path = ".../face recognition/"
face_data = []
label = []
class_id = 0
name = {}
face_cas = cv2.CascadeClassifier(".../haarcascade_frontalface_default.xml")
for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        data_item = np.load(dataset_path + fx)
        name[class_id] = fx[:-4]
        face_data.append(data_item)
        target = class_id * np.ones((data_item.shape[0],))
        class_id += 1
        label.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(label, axis=0)

print(face_dataset.shape)
print(face_labels.shape)
while True:
    res, frame = cap.read()
    if res == False:
        continue
    faces = face_cas.detectMultiScale(frame, 1.3, 5)
    for face in faces:
        x, y, w, h = face
        snip = frame[y - 10:y + h + 10, x - 10:x + w + 10]
        snip = cv2.resize(snip, (100, 100))
        out = knn(face_dataset, face_labels, snip.flatten())
        cv2.putText(frame, name[int(out)], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.imshow("faces", frame)
    pressed = cv2.waitKey(5) & 0xFF
    if pressed == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
