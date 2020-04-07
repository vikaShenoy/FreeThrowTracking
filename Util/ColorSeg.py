import cv2
import numpy as np
import matplotlib.pyplot as plt


def segment(img):
    nemo = img
    hsv_nemo = cv2.cvtColor(nemo, cv2.COLOR_BGR2HSV)
    cv2.imshow("HSV", hsv_nemo)
    cv2.waitKey(0)
    light_orange = (1, 190, 200)
    dark_orange = (18, 255, 255)
    mask = cv2.inRange(hsv_nemo, light_orange, dark_orange)
    result = cv2.bitwise_and(nemo, nemo, mask=mask)
    cv2.imshow("Frame", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    filepath = "./Data/FTVikas.mp4"
    cap = cv2.VideoCapture(filepath)
    ok, frame = cap.read()
    for i in range(30):
        ok, frame = cap.read()

    img = cv2.imread("./Data/Nemo.jpg")
    segment(frame)
