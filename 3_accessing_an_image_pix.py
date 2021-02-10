import numpy as np
import cv2

# Read the image
img = cv2.imread('pics/pictured_rocks.jpg')
print("type(img): ", type(img))
print("img.shape: ", img.shape)

# get i and j
i = int(input("i: "))
j = int(input("j: "))

# read the value at i,j
b, g, r = img[i,j]

# we could also do
b = img[i,j,0]
g = img[i,j,1]
r = img[i,j,2]

# or
b = img.item(i,j,0)
g = img.item(i,j,1)
r = img.item(i,j,2)

print(b, g, r)
