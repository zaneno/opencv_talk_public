import numpy as np
import cv2

# Read the image
img = cv2.imread('pics/pictured_rocks.jpg')

# Show the image using cv2
cv2.imshow('Now it looks better', img)

# Wait for the user to press a key
cv2.waitKey(0)

# Graceful teardown
cv2.destroyAllWindows()
