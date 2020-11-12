import cv2
import numpy as np
facedetect=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam=cv2.VideoCapture(0)
rec=cv2.face.LBPHFaceRecognizer_create()
rec.load("recognizer/trainingdata.yml")
id=0
while True:
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=facedetect.detectMultiScale(gray,1.2,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if id==1:
            id=="chakri"
        elif (id==2):
            id=="mahesh"
        cv2.putText(img,str(id),(200,300),cv2.FONT_ITALIC,255,(0,255,0),5)
    cv2.imshow("identified",img)
    if cv2.waitKey(1)==27:
        break
cam.release()
cv2.destroyAllWindows()
