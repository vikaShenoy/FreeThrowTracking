import cv2
import numpy as np
import matplotlib.pyplot as plt


def segment(img):
    nemo = img
    hsv_nemo = cv2.cvtColor(nemo, cv2.COLOR_BGR2HSV)
    light_orange = (1, 190, 200)
    dark_orange = (18, 255, 255)
    mask = cv2.inRange(hsv_nemo, light_orange, dark_orange)
    result = cv2.bitwise_and(nemo, nemo, mask=mask)
    cv2.imshow("Frame", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    img = cv2.imread("./Nemo.jpg")
    segment(img)
