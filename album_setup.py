import numpy as np
import cv2
import os, sys, pickle

album_covers_folder = 'pics/albums/covers'
album_cover_files = os.listdir(album_covers_folder)

orb = cv2.ORB_create()

list_of_albums = {}

for file in album_cover_files:
    if file[0] == '.':
        continue

    long_file = album_covers_folder + '/' + file

    album_img = cv2.imread(long_file)

    # find the keypoints with ORB
    kp = orb.detect(album_img,None)
    kp, des = orb.compute(album_img, kp)

    temp_kp = []
    temp_des = []
    for i, point in enumerate(kp):
        temp_kp.append((point.pt, point.size, point.angle, point.response, point.octave,
        point.class_id))
    album_title = input(file + " Album Title: ")
    album = {
        "file":file,
        "long_file":long_file,
        "kp":temp_kp,
        "des":des,
        "title": album_title
    }


    list_of_albums[file] = album
pickle.dump( list_of_albums, open( "pics/albums/data/list_of_albums.p", "wb" ) )
