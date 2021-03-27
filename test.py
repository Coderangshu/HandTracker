import mediapipe as mp
import cv2
import time
from faceTracker import faceTracker
from handTracker import handTracker

vc = cv2.VideoCapture(0)
# FPS Calculation Initialization
ctime = 0
ptime = 0
htracker = handTracker()
ftracker = faceTracker()
while True:
    success, img = vc.read()
    img = cv2.resize(img,(1280,720))
    htracker.findHands(img,drawHands=True)
    ftracker.findFaces(img,drawFaces=True)
    hldmkArr = htracker.getPosition()
    fldmkArr = ftracker.getPosition()
    # FPS Calculation
    ctime = time.time()
    fps = 1/(ctime-ptime)
    ptime = ctime
    cv2.putText(img,str(int(fps)),(1,25),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,255),3)
    cv2.imshow("LIVE HAND DETECTOR",img)
    cv2.waitKey(1)
