import cv2
x,y,w,h=0,0,0,0
facecascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#it is the xml file which has predifined code for trainers and detectors cascade classifier is the method to callsify what we need
#img=cv2.imread('messi5.jpg')
cap=cv2.VideoCapture(0)#capturing the video from  default cam
while True:
    ret,img=cap.read()#reading each frame
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)#we need to convert into gray for
    faces=facecascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)#facecascade object is used as multiscaledetector it takes three arguments source image
    #scale factor it means it specifies how much the image is reduced at each image scale next is minneighbor it specifies how many neighbor each candidate rectangle
    #has to be retain it it gives 4 output values for drawing the rectangle
    #print(faces)
    for (x,y,w,h) in faces:#we need to iterate the faces bcpz ther might be many faces in  the image
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)#drawing the rectangle on the faces
    l=len(faces)#counting no.of faces i]present in front of the cam  as the face of the image is stored in array in the form of pixels
    if l==0:#if the length is 0 means face is not detected which impiles he is wearing mask but here we can check only one person who is infront of the camera
        cv2.putText(img,"MASK DETECTED",(100,100),cv2.FONT_ITALIC,2,(0,0,255),2)
        print("mask  detected")
    else:
        cv2.putText(img, "MASK NOT DETECTED", (0, 100), cv2.FONT_ITALIC, 2, (0, 0, 255), 2)
        print("mask not detected")

    cv2.imshow('image',img)
    if cv2.waitKey(1)==27:
        break
cap.release()
cv2.destroyAllWindows()
