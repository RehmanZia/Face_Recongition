import os
import cv2
import numpy as np
from PIL import Image

names=[]
paths=[]

for users in os.listdir("images"):
    names.append(users)

for name in names:
    for image in os.listdir("images\{}".format(name)):
        path_string=os.path.join("images\{}".format(name),image)
        paths.append(path_string)



faces=[]
ids=[]

for img_path in paths:
    image=Image.open(img_path).convert("L")
    #image_final=image.resize((1500,1500), Image.ANTIALIAS)
    imgNP = np.array(image,"uint8")

    faces.append(imgNP)

    id=int(img_path.split("\\")[2].split("-")[0])
    print(id)
    print(imgNP)
    ids.append(id)


ids=np.array(ids)

trainer=cv2.face.LBPHFaceRecognizer_create()
trainer.train(faces,ids)

trainer.write("Trainer.yml")
print("done commpletion")
