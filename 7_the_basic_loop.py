import cv2
import numpy as np

cap = cv2.VideoCapture('pics/vids/lahore.mp4')

while(1):
    ret, frame = cap.read()
    
    cv2.imshow('frame', frame)
    k = cv2.waitKey(40) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
