import numpy as np
import cv2

# read the image
img = cv2.imread('pics/ibm_xt.jpeg',0)

# create ORB detector
orb = cv2.ORB_create()

# find the keypoints with ORB
kp = orb.detect(img,None)

# compute the descriptors with ORB
kp, des = orb.compute(img, kp)

# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)


# show the image
cv2.imshow('Feet yers!', img2)
# Wait for the user to press a key
cv2.waitKey(0)
# Graceful teardown
cv2.destroyAllWindows()
