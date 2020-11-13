import cv2
import numpy as np
cap=cv2.VideoCapture('vtest.avi')
humancas=cv2.CascadeClassifier("haarcascade_fullbody.xml")
#ret,frame1=cap.read()
#ret,frame2=cap.read()
centers=[]
distance=[]
while cap.isOpened():
    #diff=cv2.absdiff(frame1,frame1)
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    _,thresh=cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
    dilated=cv2.dilate(thresh,None,iterations=3)
    humans=humancas.detectMultiScale(gray,1.3,5)
    #contours and the social distance should be indientified in  frame 1 image
    #print(humans)
    for human in humans:
        [x,y,w,h]=human
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)
        [cx,cy]=[(2*x+w)/2,(2*y+h)/2]
        centers.append([cx,cy])
        print(centers)
        if len(centers)>=2:
            for i in range(len(centers)):
                for j in range(i + 1, len(centers)):
                    dx = centers[i][0] - centers[j][0]
                    dy = centers[i][1] - centers[j][1]
                    dist = np.sqrt(dx * dx + dy * dy)
                    #print(dist)
                    distance.append(dist)
                    if dist<=10:
                        print(dist)
                        cv2.putText(frame,"NO SOCIAL DIST ",(10,30),cv2.FONT_ITALIC,1,(255,0,0),3)
                        x,y=centers[i]

                        cv2.circle(frame,(int(x),int(y)),3,(0,255,0),2)

    cv2.imshow("socialdist",frame)
    #frame1=frame2
    #frame2=cap.read()
    centers=[]
    if cv2.waitKey(1)==27:
        break
cap.release()
cv2.destroyAllWindows()
