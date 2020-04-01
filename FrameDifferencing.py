import cv2
import numpy


def frame_differencing():
    cap = cv2.VideoCapture("./FreeThrow.mp4")
    ok, frame = cap.read()
    ok, frame2 = cap.read()
    diff = frame - frame2
    cv2.imshow("Diff", diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    frame_differencing()
