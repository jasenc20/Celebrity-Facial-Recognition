import cv2
import os
import numpy as np
from PIL import Image
import pickle

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt.xml')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_Dir = os.path.join(BASE_DIR, "images")

recognizer = cv2.face.LBPHFaceRecognizer_create()


current_id = 0
label_ids = {}
y_labels = []
x_train = []

for root, dirs, files in os.walk(image_Dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root,file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            #print(label, path)
            if label in label_ids:
                pass
            else:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]
            #print(label_ids)

            #Converts image to array
            pil_image = Image.open(path).convert("L")
            #size = (550, 550)
            #final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(pil_image, "uint8")
            #print(image_array)
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")
