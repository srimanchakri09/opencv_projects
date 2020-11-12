import os
import cv2
import numpy as np
from PIL import Image
recognizer=cv2.face.LBPHFaceRecognizer_create()
ourpath='dataset'

def getimagesnids(ourpath):
    imgpaths=[os.path.join(ourpath,f) for f in os.listdir(ourpath)]
    #print(imgpaths)
    faces=[]
    ids=[]
    for image in imgpaths:
        faceimg=Image.open(image).convert('L')
        facenp=np.array(faceimg,'uint8')
        id=os.path.split(image)[-1].split('.')[1]
        ids.append(id)
        faces.append(facenp)
        print(id)
        cv2.imshow("training",facenp)
        cv2.waitKey(10)
        return ids,faces

ids,faces=getimagesnids(ourpath)
#labelsf = np.random.randint(0, size=len(faces))
#labelsi = np.random.randint(0, size=len(ids))
#recognizer.train(faces,np.array(ids))
recognizer.save("recognizer/trainingdata.yml")
recognizer.train(faces,np.array(ids))
cv2.destroyAllWindows()
