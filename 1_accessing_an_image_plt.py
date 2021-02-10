import numpy as np
import cv2
from matplotlib import pyplot as plt

# Read the image
img = cv2.imread('pics/pictured_rocks.jpg')

# Show the image
plt.imshow(img)
plt.show()
