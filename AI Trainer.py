import cv2
import time
import PoseModule as pm
import numpy as np

detector = pm.poseDetector()
#cap = cv2.VideoCapture(0)

while True:
   # _, img = cap.read()
    img = cv2.imread("TrainerImage/biceps.jpeg")
    img = detector.findPose(img)
    cv2.imshow("Image",img)
    cv2.waitKey(1)