import cv2
import mediapipe as mp

class FaceDetection():
    def __init__(self, minDetectionConf=0.5):
        self.minDetectionConf = minDetectionConf
        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(minDetectionConf)

    def findFaces(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceDetection.process(imgRGB)
        bboxes = []
        if self.results.detections:
            for id,detection in enumerate(self.results.detections):
                print(id,detection)
                # mpDraw.draw_detection(img,detection)
                bboxC = detection.location_data.relative_bounding_box
                h,w,c = img.shape
                bbox = int(bboxC.xmin * w),int(bboxC.ymin * h), \
                       int(bboxC.width * w), int(bboxC.height * h)
                bboxes.append([bbox, detection.score])
                cv2.rectangle(img, bbox,(0,255,0),2)

        return img, bboxes

def main():
    cap = cv2.VideoCapture(0)
    detector = FaceDetection()
    while True:
        success, img = cap.read()
        img, bboxes = detector.findFaces(img)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()