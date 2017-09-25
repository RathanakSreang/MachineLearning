from imutils.video import VideoStream
from imutils import face_utils
import imutils
import argparse
import time
import cv2
import dlib
import os

ap = argparse.ArgumentParser()
ap.add_argument("-r", "--picamera", type=int, default=-1,
    help="whether or not the Raspberry Pi camera should be used")
ap.add_argument("-s", "--save", type=str, default="",
    help="folder name where use for store user images")
args = vars(ap.parse_args())

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()

print("[INFO] camera sensor warming up...")
vs = VideoStream().start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 600 pixels, and convert it to
    # grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    height, width = frame.shape[:2]
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     # detect faces in the grayayscale frame
    rects = detector(gray_frame, 0)

    # loopop over the face detections
    for rect in rects:
        (x,y,w,h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 1)

    key = cv2.waitKey(1) & 0xFF
     # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # show the frame
    cv2.imshow("Frame", frame)

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
