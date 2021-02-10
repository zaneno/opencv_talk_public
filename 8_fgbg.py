import cv2
import numpy as np
import sys

def loop_video(cap):
    # create our background subtractor object
    fgbg = cv2.createBackgroundSubtractorMOG2(
        history=50,
        varThreshold=75,
        detectShadows=True
    )

    # get the total number of frames
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # frame counter
    frame_count = 0

    # which type of frame we want to display
    cur_display = 49

    # infinite loop
    while(True):
        # loop over the video once
        while(frame_count < frames):
            # grab a frame
            ret, frame = cap.read()

            # get the foreground mask for the frame
            fgmask = fgbg.apply(frame)

            # blur the mask
            fgmask = cv2.GaussianBlur(fgmask, (25,25), 0)

            # apply the foreground mask to the frame
            res = cv2.bitwise_and(frame, frame, mask=fgmask)

            # our display options
            displays = {49: frame, 50: fgmask, 51: res}

            cv2.imshow('frame', displays[cur_display])

            # wait a while for user input
            k = cv2.waitKey(40) & 0xff
            # ESC
            if k == 27:
                return
            # change display
            elif k in displays:
                cur_display = k

            frame_count += 1

        # reset video to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = 0


# set up a video capture
cap = cv2.VideoCapture('pics/vids/cars.mp4')

loop_video(cap)

cap.release()
cv2.destroyAllWindows()
