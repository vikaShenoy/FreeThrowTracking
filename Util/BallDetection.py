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
        width_height = (2*r) + BOX_PADDING
        return (x-r, y-r, width_height, width_height)
    else:
        print("No circles found.")
        return None


def apply_morphological_operators(frame):
    """."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (19, 19), 0)

    block_size = 11
    thresh = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, 1)

    kernel = np.ones((3, 3), np.uint8)

    opening = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    closing = cv2.morphologyEx(
        opening, cv2.MORPH_CLOSE, kernel, iterations=4)

    return closing


def basketball_candidate(contour):
    """Apply filters to a contour to detect whether it's a basketball."""
    area = cv2.contourArea(contour)
    if area <= 600 or area >= 1000:
        return False
    return True


def morphological_detector(cap):
    """Take in a video and read frames until a ball which meets all the filters is detected.
    Return the bounding box around the detected ball."""
    detected = False

    valid_contours = []

    while not detected:
        print(1)
        ok, frame = cap.read()

        if not ok:
            return None

        filtered_frame = apply_morphological_operators(frame)

        contours, hierarchy = cv2.findContours(
            filtered_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(len(contours))
        for contour in contours:
            valid_contour = basketball_candidate(contour)
            if valid_contour:
                detected = True
                valid_contours.append(contour)

    cv2.drawContours(frame, valid_contours, -1, (255, 0, 0), 2)
    show(frame)

    return (0, 0, 0, 0)


def show(img):
    """Test function to show opencv2 images."""
    cv2.imshow("Test", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    filepath = "./Data/FTVikas.mp4"
    cap = cv2.VideoCapture(filepath)
    bbox = morphological_detector(cap)
    print(f"Bbox: {bbox}")
