import cv2
import numpy as np
face_detect=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam=cv2.VideoCapture(0)
id=input('enter the user id ')
samplenum=0
while (True):
    ret,img=cam.read();
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
    faces=face_detect.detectMultiScale(gray,1.5,5);
    for (x,y,w,h) in faces:
        samplenum=samplenum+1
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3);
        cv2.imwrite("dataset/user"+str(id)+"."+str(samplenum)+".jpg",gray[y-10:y+h+10,x-10:x+w+10])
    cv2.imshow("image",img)
    cv2.waitKey(1)
    if samplenum>20:
        break
cam.release()
cv2.destroyAllWindows()
