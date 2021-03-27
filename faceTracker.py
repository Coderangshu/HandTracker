import mediapipe as mp
import cv2
import time

class faceTracker():
    def __init__(self,mode=False,maxFaces=1,minDetectionConfidence=0.5,minTrackingConfidence=0.5):
        self.mode = mode
        self.maxFaces = maxFaces
        self.minDetectionConfidence = minDetectionConfidence
        self.minTrackingConfidence = minTrackingConfidence
        self.face = mp.solutions.face_mesh.FaceMesh(self.mode,self.maxFaces,self.minDetectionConfidence,self.minTrackingConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    def findFaces(self,img, drawFaces = False, drawConnections = True):
        self.img = img
        imgRGB = cv2.cvtColor(self.img,cv2.COLOR_BGR2RGB)
        self.results = self.face.process(imgRGB)
        if self.results.multi_face_landmarks and drawFaces:
            for faceLMS in self.results.multi_face_landmarks:
                if drawConnections:
                    self.mpDraw.draw_landmarks(img,faceLMS,mp.solutions.face_mesh.FACE_CONNECTIONS)
                else:
                    self.mpDraw.draw_landmarks(img,faceLMS)

    def getPosition(self, noOfFaces = True, markId = 469):
        getPositionArr = []
        if self.results.multi_face_landmarks:
            for faceId,faceLMS in enumerate(self.results.multi_face_landmarks):
                for id,lm in enumerate(faceLMS.landmark):
                    h,w,c = self.img.shape
                    cx, cy = int(lm.x*w),int(lm.y*h)
                    if id==markId:
                        cv2.circle(self.img,(cx,cy),15,(0,0,255),cv2.FILLED)
                    getPositionArr.append((faceId,id,cx,cy))
        if noOfFaces:
            cv2.putText(self.img,"Face = "+str(len(getPositionArr)//468),(1100,25),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
        return getPositionArr

def main():

    vc = cv2.VideoCapture(0)
    # FPS Calculation Initialization
    ctime = 0
    ptime = 0
    tracker = faceTracker()
    while True:
        success, img = vc.read()
        img = cv2.resize(img,(1280,720))
        tracker.findFaces(img,drawFaces=True)
        ldmkArr = tracker.getPosition()
        # FPS Calculation
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime
        cv2.putText(img,str(int(fps)),(1,25),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),3)
        cv2.imshow("LIVE FACE DETECTOR",img)
        cv2.waitKey(1)

if __name__=="__main__":
    main()