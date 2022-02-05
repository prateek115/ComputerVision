import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection()

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    print(results)

    if results.detections:
        for id,detection in enumerate(results.detections):
            print(id,detection)
            # mpDraw.draw_detection(img,detection)
            bboxC = detection.location_data.relative_bounding_box
            h,w,c = img.shape
            bbox = int(bboxC.xmin * w),int(bboxC.ymin * h), \
                   int(bboxC.width * w), int(bboxC.height * h)

            cv2.rectangle(img, bbox,(0,255,0),2)


    cv2.imshow("Image", img)
    cv2.waitKey(1)

