import mediapipe as mp
import cv2
import time

class handTracker():
    def __init__(self,mode=False,maxHands = 2, minDetectionConfidence = 0.5, minTrackingConfidence = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackingConfidence = minTrackingConfidence
        self.hands = mp.solutions.hands.Hands(self.mode,self.maxHands,self.minDetectionConfidence,self.minTrackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self,img, drawHands = False, drawConnections = True):
        self.img = img
        imgRGB = cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks and drawHands:
            for handLMS in self.results.multi_hand_landmarks:
                if drawConnections:
                    self.mpDraw.draw_landmarks(img,handLMS,mp.solutions.hands.HAND_CONNECTIONS)
                else:
                    self.mpDraw.draw_landmarks(img,handLMS)

    def getPosition(self, noOfHands = True, markId = 22):
        getPositionArr = []
        if self.results.multi_hand_landmarks:
            for handId,handLMS in enumerate(self.results.multi_hand_landmarks):
                for id,lm in enumerate(handLMS.landmark):
                    # print(handId,id,lm)
                    h,w,c = self.img.shape
                    cx, cy = int(lm.x*w),int(lm.y*h)
                    # print(handId,id,cx,cy)
                    if id==markId:
                        cv2.circle(self.img,(cx,cy),15,(0,0,255),cv2.FILLED)
                    getPositionArr.append((handId,id,cx,cy))
        if noOfHands:
            cv2.putText(self.img,"Hand = "+str(len(getPositionArr)//21),(1100,55),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
        return getPositionArr


def main():

    vc = cv2.VideoCapture(0)
    # FPS Calculation Initialization
    ctime = 0
    ptime = 0
    tracker = handTracker()
    while True:
        success, img = vc.read()
        img = cv2.resize(img,(1280,720))
        tracker.findHands(img,drawHands=True)
        ldmkArr = tracker.getPosition()
        # FPS Calculation
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime
        cv2.putText(img,str(int(fps)),(1,25),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),3)
        cv2.imshow("LIVE HAND DETECTOR",img)
        cv2.waitKey(1)

if __name__=="__main__":
    main()
