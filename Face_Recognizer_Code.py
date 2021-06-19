import cv2
import os
from djitellopy import tello


faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video_capture = cv2.VideoCapture(0)

# Call the trained model yml file to recognize faces
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Trainer1.yml")

names = []
for users in os.listdir("images"):
    names.append(users)

while True:

    _, img = video_capture.read()
    img=cv2.resize(img,(500,500))
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray_image, scaleFactor=1.1, minNeighbors=4
    )

    # Try to predict the face and get the id
    # Then check if id == 1 or id == 2
    # Accordingly add the names
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, conf = recognizer.predict(gray_image[y : y + h, x : x + w])
        print(id)
        if conf>90:
            cv2.putText(
                img,
                "Unknown",
                (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

        else:
            cv2.putText(
                img,
                names[id - 1],
                (x, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )


    cv2.imshow("Recognize", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# video_capture.release()
cv2.destroyAllWindows()