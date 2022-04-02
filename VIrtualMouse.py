import cv2
import numpy as np
import time
import handTrackingModule as htm
import autopy

wCam,hCam = 640,680
pTime=0
plocx, plocy =0, 0
clocx, clocy = 0,0
frameR = 100
smooth = 7
wScr, hScr = autopy.screen.size()

cap = cv2.VideoCapture(0)
detector = htm.DetectorForHands(maxHands=1)
cap.set(3,wCam)
cap.set(4,hCam)

while True:

    _,img = cap.read()
    img = detector.findHands(img)
    lmList,bbox = detector.findpos(img,draw=False)

    if len(lmList) != 0:
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]

        fingres = detector.fingerUp()
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 0), 2)

        # checking if only index is up : for moving mouse
        if fingres[1] == 1 and fingres[2] == 0:

            # scaling the values according to screen size
            x3 = np.interp(x1,(frameR,wCam-frameR),(0,wScr))
            y3 = np.interp(y1,(frameR,hCam-frameR),(0,hScr))

            #Conversions for smooth transitions of cursor
            clocx = plocx + (x3-plocx) / smooth
            clocy = plocy + (y3-plocy) / smooth

            autopy.mouse.move(wScr-clocx,clocy)
            cv2.circle(img,(x1,y1),12,(0,0,255),cv2.FILLED)
            plocx, plocy = clocx, clocy

        # Checking if index and middle finger is up: For clicking
        if fingres[1] == 1 and fingres[2] == 1:
            length, img , lineInfo = detector.findDistance(8,12,img)
            print(length)

            if length <40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 12, (0, 125, 0), cv2.FILLED)
                autopy.mouse.click()

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (225, 0, 0), 3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)

