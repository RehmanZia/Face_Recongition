import cv2
import numpy as np
import face_recognition
import os
import time



path = "TestNew_Method"

images = []
Names = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    Names.append(os.path.splitext(cl)[0])

print(Names)


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

start= time.perf_counter()
know=findEncodings(images)
print(start)
print("Encoding Completed")

cap = cv2.VideoCapture(0)

while True:
    success,img = cap.read()
    imgs= cv2.resize(img,(0,0),None,0.25,0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgs)
    encodeCurFrame = face_recognition.face_encodings(imgs,facesCurFrame)

    for encodeface,faceloc in zip(encodeCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(know,encodeface)
        faceDis = face_recognition.face_distance(know,encodeface)
        #print(faceDis)
        matchIndex=np.argmin(faceDis)

        if faceDis[matchIndex] < 0.50:
            name = Names[matchIndex].upper()
        else:
            name = 'Unknown'
        y1, x2, y2, x1 = faceloc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, name, (x1 + 6, y1 - 25), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, "Yes", (x1 + 6, y1 - 5), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Image",img)
    cv2.waitKey(1)

