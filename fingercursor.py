import cv2
import numpy as np
import pyautogui as pg
pg.FAILSAFE=False
r_area=[100,1700]
showCentroid=False
center=()
rx,ry=0,0
cx,cy=0,0
def swap( array, i, j):
	temp = array[i]
	array[i] = array[j]
	array[j] = temp

#def drawCentroid(vid, color_area, mask, showCentroid):

 #           return center
  #  else:
        # return error handling values
   #     return (-1, -1)
finalcontours=np.array([],np.uint8)
cap=cv2.VideoCapture(0)

kernal=np.ones((5,5),np.uint8)
while True:
    ret,frame=cap.read()
    gblur=cv2.GaussianBlur(frame,(5,5),0)
    hsv=cv2.cvtColor(gblur,cv2.COLOR_BGR2HSV)
    #gblur=cv2.GaussianBlur(hsv,(5,5),0)
    lb=np.array([164,113,120])
    ub=np.array([255,255,255])
    mask1=cv2.inRange(hsv,lb,ub)
    res=cv2.bitwise_and(frame,frame,mask=mask1)
    _,mask=cv2.threshold(res,50,255,cv2.THRESH_BINARY_INV)
    morphed=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernal)
    #gblur=cv2.GaussianBlur(morphed,(5,5),0)
    cannied = cv2.Canny(morphed, 100, 200)
    contour, _ = cv2.findContours(cannied, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    l = len(contour)
    area = np.zeros(l)

    # filtering contours on the basis of area rane specified globally
    for i in range(l):
        if cv2.contourArea(contour[i]) > r_area[0] and cv2.contourArea(contour[i]) < r_area[1]:
            area[i] = cv2.contourArea(contour[i])
        else:
            area[i] = 0

    a = sorted(area, reverse=True)

    # bringing contours with largest valid area to the top
    for i in range(l):
        for j in range(1):
            if area[i] == a[j]:
                swap(contour, i, j)

    if l > 0:
        # finding centroid using method of 'moments'
        M = cv2.moments(contour[0])
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            center = (cx, cy)
            if showCentroid:
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
    print(center)
    rx=3*cx
    ry=2.25*cy
    print(rx,ry)
    pg.moveTo(rx,ry,duration=0.2)
    #cv2.imshow('blured',gblur)
    cv2.imshow("contoured",frame)
    #cv2.imshow('morphed',morphed)
    if cv2.waitKey(100) == 27:
        break
cap.release()
cv2.destroyAllWindows()
