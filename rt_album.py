import numpy as np
import cv2
import matplotlib.pyplot as plt
import os, sys
import pickle

def load_album_covers(pickle_file = "pics/albums/data/list_of_albums.p"):
    """
    Read album cover info from pickle file
    """
    list_of_albums = pickle.load(open(pickle_file, "rb"))
    for file, album in list_of_albums.items():
            for j, point in enumerate(album['kp']):
                list_of_albums[file]['kp'][j] = cv2.KeyPoint(
                    x=point[0][0],
                    y=point[0][1],
                    _size=point[1],
                    _angle=point[2],
                    _response=point[3],
                    _octave=point[4],
                    _class_id=point[5]
                )
    return list_of_albums

def get_total_matches(kp1, des1, img2):
    """
    Returns the total matches.
    compares a keypoint, descriptor pair with image.
    Only returns matches of a certain quality.
    """
    # find the keypoints with ORB
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

    # ratio test as per Lowe's paper
    total_matches = 0
    for i, pair in enumerate(matches[:min(len(des1),len(des2))-1]):
        if len(pair) == 2 and pair[0].distance < 0.7*pair[1].distance:
            total_matches += 1

    return total_matches

def find_album(img, album_covers):
    """
    Search through a provided dict of album covers
    for any that are present in a given image.

    Returns the album dict and the number of
    keypoints it was able to match in the image.
    """
    best_match = 0
    best_mask = None
    best_file = None

    for album in album_covers.values():
        total_matches = get_total_matches(
            album['kp'],
            album['des'],
            img
        )
        # print(total_matches)
        if type(total_matches) == int and total_matches > best_match:
            best_match = total_matches
            best_album = album

    if best_album is not None:
        return best_album, best_match
    return None, 0

def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

def loop_video(cap):
    # get the total number of frames
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # frame counter
    frame_count = 0

    cur_album = None

    # infinite loop
    while(True):
        # loop over the video once
        while(frame_count < frames):
            # grab a frame
            ret, frame = cap.read()

            # why did you film this in portrait mode??!!
            #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame = cv2.resize(frame, (506,900))

            # just check for albums every thirty frames
            if frame_count % 30 == 0:
                album, matches = find_album(frame, album_covers)
                #print(album, matches)

            # if we found an album, say so
            # we want more than 20 matches to be sure
            if album is not None and matches > 20:
                cv2.putText(
                    frame,
                    album["title"],
                    org = (10,50),
                    fontFace = cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    thickness = 2,
                    color = (255, 0, 0),
                    lineType=2
                )
            cv2.imshow('frame', frame)


            # wait a while for user input
            k = cv2.waitKey(30) & 0xff
            # ESC
            if k == 27:
                return
            elif k == 110:
                cv2.imwrite('pics/saved_frames/frame'+str(frame_count)+'.jpg', frame)

            frame_count += 1

        # reset video to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = 0

# Initiate ORB detector
orb = cv2.ORB_create()

# laod our pickled album cover info
album_covers = load_album_covers()

# set up a video capture
cap = cv2.VideoCapture(sys.argv[1])

# start the loop
loop_video(cap)

cap.release()
cv2.destroyAllWindows()
