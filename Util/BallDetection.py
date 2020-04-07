"""Utility functions to detect a ball in an image."""

import cv2
import numpy as np


def hough_detector(frame):
    """Take in a frame from a video. Detect the basketball in the image and
    return a bounding box around its location which can be used for tracking."""
    blur = cv2.GaussianBlur(frame, (9, 9), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(image=gray, method=cv2.HOUGH_GRADIENT,
                               dp=1, minDist=200, param1=175, param2=20, minRadius=10, maxRadius=20)

    if circles is not None:
        # Circles are 'doublewrapped' in an extra list? Not sure what's going on, this seems to work.
        circles = circles[0]
        print(f"Circles found: {len(circles)}")
        # Taking the smallest circle here.
        (x, y, r) = min(circles, key=lambda x: math.pi * (x[2]**2))
        RADIUS = r
        # Add some tolerance around the ball
        widthHeight = (2*r) + BOX_PADDING
        return (x-r, y-r, widthHeight, widthHeight)
    else:
        print("No circles found.")
        return None


def hierarchical_detector(cap):
    """Take in a video and read frames until a ball which meets all the filters is detected.
    Return the bounding box around the detected ball."""
    ok, frame = cap.read()

    return (0, 0, 0, 0)


def show(img):
    """Test function to show opencv2 images."""
    cv2.imshow("Test", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    filepath = "./Data/JAllen.mp4"
    cap = cv2.VideoCapture(filepath)
    bbox = hierarchical_detector(cap)
    print(f"Bbox: {bbox}")
