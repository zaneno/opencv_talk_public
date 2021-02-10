import numpy as np
import cv2

# Read the image
img = cv2.imread('pics/flowers/jpg/image_00405.jpg')

# blur the image
blur = cv2.blur(img, (5,5))

# blur the image with a larger kernel
blurrier = cv2.blur(img, (25,25))

# blur the image with an even larger kernel
blurriest = cv2.blur(img, (100,100))

# stack the images together
h_stack = np.vstack((
    np.hstack((img, blur)),
    np.hstack((blurrier, blurriest))
))

# show the stacked images
# and wair for keypress
cv2.imshow("Blurry", h_stack)
cv2.waitKey(0)
cv2.destroyAllWindows()

# blur using bilateral filter
b_blur = cv2.bilateralFilter(img, 9, 75, 75)

# blur using median blur
m_blur = cv2.medianBlur(img, 5)

# blur using gaussian blur
g_blur = cv2.GaussianBlur(img, (25,25), 0)

# stack the images together
h_stack = np.vstack((
    np.hstack((img, b_blur)),
    np.hstack((m_blur, g_blur))
))

# show the stacked images
# and wair for keypress
cv2.imshow("Blurry", h_stack)
cv2.waitKey(0)
cv2.destroyAllWindows()
