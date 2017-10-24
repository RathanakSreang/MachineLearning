from imutils import face_utils
from imutils.face_utils import FaceAligner
import imutils
import argparse
import time
import cv2
import dlib
import os

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save", type=str, default="",
    help="folder name where use for store user images", required=True)
ap.add_argument("-p", "--path", type=str, default="",
    help="folder name where use for store user images", required=True)
args = vars(ap.parse_args())

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
fa = FaceAligner(predictor, desiredFaceWidth=256)

count_image = 0
save_file_name = args["save"]
base_dir = "images/"

if save_file_name:
    base_dir = base_dir + save_file_name + "/"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

valid_images = [".jpg",".gif",".png"]
path = args["path"]
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue

    frame = cv2.imread(os.path.join(path,f))
    # frame = imutils.resize(frame, width=600)
    height, width = frame.shape[:2]
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayayscale frame
    rects = detector(gray_frame, 0)

    if len(rects) > 0:
        faceAligned = fa.align(frame, gray_frame, rects[0])
        image_name = base_dir + str(count_image) + ".png"
        # save image
        cv2.imwrite(image_name, faceAligned)
        # show image
        cv2.imshow(image_name, faceAligned)
        count_image += 1
        if count_image > 10:
            count_image = 0

    # loopop over the face detections
    for rect in rects:
        (x,y,w,h) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 1)

    # show the frame
    cv2.imshow("Frame", frame)

cv2.waitKey(0)

# do a bit of cleanup
cv2.destroyAllWindows()
