import cv2
import time
import os
import handTrackingModule as htm

wCam, hCam = 640, 480
pTime=0

detector = htm.DetectorForHands(detectionConf=0.7)

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

filepath = "FingerCount"
myList = os.listdir(filepath)
overlayList = []
for impath in myList:
    image = cv2.imread(f'{filepath}/{impath}')
    overlayList.append(image)
print(len(overlayList))

tipIds = [4, 8, 12, 16, 20]
while True:
    _, img = cap.read()

    img = detector.findHands(img)
    lmList = detector.findpos(img,draw=False)

    if len(lmList) != 0:
        fingers = []

        if lmList[tipIds[0]][1] < lmList[tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1,5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        totalfingers = fingers.count(1)
        h,w,c = overlayList[totalfingers].shape
        img[0:h, 0:w] = overlayList[totalfingers]

        cv2.rectangle(img, (0,270),(150,480),(0,255,0), cv2.FILLED)
        cv2.putText(img, str(totalfingers), (35,420), cv2.FONT_HERSHEY_COMPLEX, 5, (255,0,0), 25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (500, 450), cv2.FONT_HERSHEY_PLAIN, 3, (225, 0, 0), 3)
    cv2.imshow("Img", img)
    cv2.waitKey(1)
