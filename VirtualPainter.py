import numpy as np
import cv2
import time
import os
import handTrackingModule as htm

folder = "VirtualMenu"
mylist = os.listdir(folder)
overlayList = []
xp,yp = 0,0
eraser = 80
imgCanvas = np.zeros((720,1280,3), np.uint8)

for img in mylist:
    image = cv2.imread(f'{folder}/{img}')
    overlayList.append(image)


header = overlayList[0]
drawColor = (255,0,255)
cap = cv2.VideoCapture(0)
cap.set(3,1579)
cap.set(4,165)

detector = htm.DetectorForHands(detectionConf=0.7)

while True:

    _, img = cap.read()
    img = cv2.flip(img,1)

    img = detector.findHands(img)
    lmList = detector.findpos(img,draw=False)

    if len(lmList) != 0:

        x1,y1 = lmList[8][1:]          #index finger
        x2,y2 = lmList[12][1:]         #middle finger

        fingres = detector.fingerUp()
        print(fingres)
        #selection code
        if fingres[1] and fingres[2]:
            print("Selection")
            xp, yp = 0, 0
            if y1 < 135:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (0,0,255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255,0,0)
                elif 800 < x1 < 1000:
                    header = overlayList[2]
                    drawColor = (0,125,0)
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0,0,0)
            cv2.rectangle(img, (x1, y1 - 30), (x2, y2 + 30), drawColor, cv2.FILLED)

        #Drawing tool
        if fingres[1] and fingres[2] == False:
            cv2.circle(img,(x1,y1),15,drawColor,cv2.FILLED)
            print("Drawing")
            if xp == 0 and yp == 0:
                xp,yp = x1,y1

            if drawColor == (0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraser)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraser)
            else:
                cv2.line(img,(xp,yp),(x1,y1),drawColor,15)
                cv2.line(imgCanvas,(xp,yp),(x1,y1),drawColor,15)
            xp , yp = x1 , y1

    imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)

    img[0:134,0:1280] = header
    img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)
    cv2.imshow("Image", img)
    cv2.waitKey(1)