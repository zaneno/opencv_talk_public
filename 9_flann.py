import numpy as np
import cv2
import matplotlib.pyplot as plt
import os, sys

def get_matches_mask(img1, img2):
    # load images
    img1 = cv2.imread(img1,cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2,cv2.IMREAD_GRAYSCALE)

    # Initiate SIFT detector
    orb = cv2.ORB_create()
    
    # find the keypoints with ORB
    kp1 = orb.detect(img1,None)
    kp1, des1 = orb.compute(img1, kp1)

    kp2 = orb.detect(img2,None)
    kp2, des2 = orb.compute(img2, kp2)

    # FLANN parameters
    FLANN_INDEX_LSH = 6
    index_params= dict(algorithm = FLANN_INDEX_LSH,
                       table_number = 6, # 12
                       key_size = 12,     # 20
                       multi_probe_level = 1) #2
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    try:
        matches = flann.knnMatch(des1,des2,k=2)
    except cv2.error as e:
        return [], 0
    
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    
    # ratio test as per Lowe's paper
    total_matches = 0
    for i, pair in enumerate(matches[:min(len(des1),len(des2))-1]):
        if len(pair) == 2 and pair[0].distance < 0.7*pair[1].distance:
            matchesMask[i]=[1,0]
            total_matches += 1

    return matchesMask, total_matches

pics_folder = sys.argv[2]
compare_files = os.listdir(pics_folder)

artist_file = sys.argv[1]

best_match = 0
best_mask = None
best_file = None
for other_file in compare_files:
    if other_file[0] == '.' or pics_folder+other_file == artist_file:
        continue
    other_file = pics_folder + other_file
    matchesMask, total_matches = get_matches_mask(other_file, artist_file)
    if total_matches > best_match:
        best_match = total_matches
        best_mask = matchesMask
        best_file = other_file

print(artist_file)
print(best_file)

img1 = cv2.imread(artist_file,cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(best_file,cv2.IMREAD_GRAYSCALE)

orb = cv2.ORB_create()
# find the keypoints with ORB
kp1 = orb.detect(img1,None)
kp1, des1 = orb.compute(img1, kp1)

kp2 = orb.detect(img2,None)
kp2, des2 = orb.compute(img2, kp2)


# FLANN parameters
FLANN_INDEX_LSH = 6
index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2

search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=1)

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = best_mask,
                   flags = cv2.DrawMatchesFlags_DEFAULT)
img1 = cv2.imread(artist_file)
img2 = cv2.imread(best_file)
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
cv2.imshow('frame', img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
