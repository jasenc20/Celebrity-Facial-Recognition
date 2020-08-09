import numpy as np
import cv2
import pickle
import tkinter

# frontal face cascade
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

scarlet_app = 0
robert_app = 0
benedict_app = 0
chadwick_app = 0
chrise_app = 0
chrish_app = 0
chrisp_app = 0
mark_app = 0
tom_app = 0


labels = {}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture('video/scarlet-test.mp4')

while(True):
    # capture frames
    ret, frame = cap.read()

    # convert frame to grey
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)

    i = 80
    for(x, y, w, h) in faces:
        print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Image recognizer
        id_, conf = recognizer.predict(roi_gray)
        if conf <= 90:
            print(id_)
            print(labels[id_])
            if(labels[id_ ] == "scarlet-johansson"):
                scarlet_app = scarlet_app + 1
            elif(labels[id_] == "benedict-cumberbatch"):
                benedict_app = benedict_app + 1
            elif (labels[id_] == "chadwick-boseman"):
                chadwick_app = chadwick_app + 1
            elif (labels[id_] == "chris-evans"):
                chrise_app = chrise_app + 1
            elif (labels[id_] == "chris-hemsworth"):
                chrish_app = chrish_app + 1
            elif (labels[id_] == "chris-pratt"):
                chrisp_app = chrisp_app + 1
            elif (labels[id_] == "mark-ruffalo"):
                mark_app = mark_app + 1
            elif (labels[id_] == "robert-downey-jr"):
                robert_app = robert_app + 1
            elif (labels[id_] == "tom-holland"):
                tom_app = tom_app + 1


            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

        #name = "my.jpg"
        #img_item = name
        #cv2.imwrite(img_item, roi_gray)

        color = (255, 0, 0)
        stroke = 2
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)


    # display video
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

if(scarlet_app > 1500):
    print("Scarlet Johansson is in this clip")
if(tom_app > 1500):
    print("Tom Holland is in this clip")
if(benedict_app > 1500):
    print("Bennedict Cumberbatch is in this clip")
if(chadwick_app > 1500):
    print("Chadwick-Boseman is in this clip")
if(chrise_app > 1500):
    print("Chris Evans is in this clip")
if(chrish_app > 1500):
    print("Chris Hemsworth is in this clip")
if(chrisp_app > 1500):
    print("Mark Ruffalo is in this clip")
if(robert_app > 1500):
    print("Robert Downey jr is in this clip")

cap.release()
cv2.destroyAllWindows()