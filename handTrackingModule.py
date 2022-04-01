import cv2
import mediapipe as mp
import time


class DetectorForHands:
    def __init__(self, mode=False, maxHands=2, complexity=1, detectionConf=0.5, trackingConf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionConf = detectionConf
        self.trackingConf = trackingConf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.complexity, self.detectionConf, self.trackingConf)
        self.mpDraw = mp.solutions.drawing_utils

        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for landmarks in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(img, landmarks, self.mpHands.HAND_CONNECTIONS)

        return img

    def findpos(self, img, handNo=0, draw=True):

        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h,w,c = img.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id,cx,cy])
                if draw:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 0), cv2.FILLED)

        return self.lmList

    def fingerUp(self):

        fingers = []

        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

def main():
    pTime = 0
    cTime = 0
    cam = cv2.VideoCapture(0)
    detector = DetectorForHands()

    while True:
        success, img = cam.read()
        img = detector.findHands(img)
        lmList = detector.findpos(img)
        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (225, 0, 0), 2)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
