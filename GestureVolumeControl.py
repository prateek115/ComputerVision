import cv2
import time
import numpy as np
import handTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


wCam, hCam = 640, 480

detector = htm.DetectorForHands(detectionConf=0.7)

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
vol=0
volRec=400
volPer=0

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()

minVol = volRange[0]
maxVol = volRange[1]

while True:
    _, img = cap.read()
    detector.findHands(img)
    lmlist = detector.findpos(img, draw=False)
    if len(lmlist) != 0:
        #print(lmlist[4],lmlist[8])
        x1, y1 = lmlist[4][1], lmlist[4][2]
        x2, y2 = lmlist[8][1], lmlist[8][2]

        cx,cy= (x1+x2)//2, (y1+y2)//2

        cv2.circle(img, (x1,y1), 12, (255,0,255), cv2.FILLED)
        cv2.circle(img, (x2,y2), 12, (255,0,255), cv2.FILLED)
        cv2.circle(img, (cx,cy), 12, (255,0,255), cv2.FILLED)
        cv2.line(img, (x1,y1),(x2,y2), (255,0,255),3)

        length = math.hypot(x2-x1, y2-y1)
        vol = np.interp(length,[25,160],[minVol,maxVol])
        volRec = np.interp(length,[25,160],[400,150])
        volPer = np.interp(length,[25,160],[0,100])
        print(vol)
        volume.SetMasterVolumeLevel(vol, None)
        if length<25:
            cv2.circle(img, (cx, cy), 12, (0,255,0), cv2.FILLED)

    cv2.rectangle(img,(50,150), (85,400),(0,255,0),3)
    cv2.rectangle(img,(50,int(volRec)), (85,400),(0,255,0),cv2.FILLED)
    cv2.putText(img, f'{int(volPer)}%', (40, 45 0), cv2.FONT_HERSHEY_PLAIN, 3, (225, 0, 0), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (225, 0, 0), 3)
    cv2.imshow("Img",img)
    cv2.waitKey(1)