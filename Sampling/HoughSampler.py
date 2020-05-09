# hough_circle.py

import cv2
import numpy as np


def nothing(x):
    # We need a callback for the createTrackbar function.
    # It doesn't need to do anything, however.
    pass


# img_original = cv2.imread('../images/coloured_balls.jpg')
cap = cv2.VideoCapture("./Data/FTGordon.mp4")

for i in range(90):
    ret, img_original = cap.read()


ret, img_original = cap.read()

blur = cv2.GaussianBlur(img_original, (9, 9), 0)
# Convert the image to grayscale for processing
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

# Create the GUI elements
cv2.namedWindow('Hough Circle Transform')
cv2.createTrackbar('Canny Threshold', 'Hough Circle Transform', 1, 500,
                   nothing)
cv2.createTrackbar('Accumulator Threshold', 'Hough Circle Transform', 1, 500,
                   nothing)
cv2.createTrackbar("Min Radius", 'Hough Circle Transform', 0, 100, nothing)
cv2.createTrackbar("Max Radius", 'Hough Circle Transform', 1, 100, nothing)
# Set some default parameters
cv2.setTrackbarPos("Max Radius", 'Hough Circle Transform', 100)
cv2.setTrackbarPos("Canny Threshold", 'Hough Circle Transform', 100)
cv2.setTrackbarPos("Accumulator Threshold", 'Hough Circle Transform', 20)

while True:
    # Read the parameters from the GUI
    param1 = cv2.getTrackbarPos('Canny Threshold', 'Hough Circle Transform')
    param2 = cv2.getTrackbarPos('Accumulator Threshold',
                                'Hough Circle Transform')
    minRadius = cv2.getTrackbarPos('Min Radius', 'Hough Circle Transform')
    maxRadius = cv2.getTrackbarPos('Max Radius', 'Hough Circle Transform')

    # Attempt to detect circles in the grayscale image.
    circles = cv2.HoughCircles(gray,
                               cv2.HOUGH_GRADIENT,
                               1,
                               120,
                               param1=param1,
                               param2=param2,
                               minRadius=minRadius,
                               maxRadius=maxRadius)

    # Create a new copy of the original image and draw the detected circles on it.
    img = img_original.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            # Draw the outer circle
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow('Hough Circle Transform', img)

    if cv2.waitKey(100) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
