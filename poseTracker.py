import mediapipe as mp
import cv2
import time

class poseTracker():
    def __init__(self,mode=False,upperBody=False,smoothLandmarks=True, minDetectionConfidence = 0.6, minTrackingConfidence = 0.6):
        self.mode = mode
        self.upperBody = upperBody
        self.smoothLandmarks = smoothLandmarks
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackingConfidence = minTrackingConfidence
        self.pose = mp.solutions.pose.Pose(self.mode,self.upperBody,self.minDetectionConfidence,self.minTrackingConfidence,self.smoothLandmarks)
        self.mpDraw = mp.solutions.drawing_utils

    def detectPose(self,img, drawPose = False, drawConnections = True):
        self.img = img
        imgRGB = cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks and drawPose:
            if drawConnections:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,mp.solutions.pose.POSE_CONNECTIONS)
            else:
                self.mpDraw.draw_landmarks(img,self.results.pose_landmarks)

    def getPosition(self, noOfHands = True, markId = 33):
        getPositionArr = []
        if self.results.pose_landmarks:
            if self.results.pose_landmarks:
                for id,lm in enumerate(self.results.pose_landmarks.landmark):
                    # print(handId,id,lm)
                    h,w,c = self.img.shape
                    cx, cy = int(lm.x*w),int(lm.y*h)
                    # print(handId,id,cx,cy)
                    if id==markId:
                        cv2.circle(self.img,(cx,cy),15,(0,0,255),cv2.FILLED)
                    getPositionArr.append((id,cx,cy))
        # if noOfHands:
        #     cv2.putText(self.img,"Hand = "+str(len(getPositionArr)//21),(1100,55),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
        return getPositionArr

def main():

    vc = cv2.VideoCapture(0)
    # FPS Calculation Initialization
    ctime = 0
    ptime = 0
    tracker = poseTracker()
    while True:
        success, img = vc.read()
        img = cv2.resize(img,(1280,720))
        tracker.detectPose(img,drawPose=True)
        ldmkArr = tracker.getPosition()
        # FPS Calculation
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime
        cv2.putText(img,str(int(fps)),(1,25),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),3)
        cv2.imshow("LIVE POSE DETECTOR",img)
        cv2.waitKey(1)

if __name__=="__main__":
    main()
