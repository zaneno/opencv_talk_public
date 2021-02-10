import numpy as np
import cv2
from matplotlib import pyplot as plt

# Read the image
img = cv2.imread('pics/flowers/jpg/image_00405.jpg')
rows, cols, depth = img.shape

# for every row
for i in range(rows):
    # for every column
    for j in range(cols):
        # get the pixel
        b, g, r = img[i,j]

        # minimum acceptable distance from green
        min_dist = 220

        # calculate the euclidean distance between this pixel and green
        color_dist = np.linalg.norm(np.array((r,g,b))-np.array((0,255,0)))

        # if this color is too close to green
        if color_dist < min_dist:
            # set this pixel to black
            img.itemset((i,j,0),0)
            img.itemset((i,j,1),0)
            img.itemset((i,j,2),0)
    cv2.imshow('Less Green', img)
    cv2.waitKey(1)

cv2.destroyAllWindows()

# show the image
cv2.imshow('No Green!', img)

# Wait for the user to press a key
cv2.waitKey(0)

# Graceful teardown
cv2.destroyAllWindows()
