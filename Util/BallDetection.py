"""Detect a basketball in an image. First step in ball tracking."""

import cv2
import numpy as np
import math


def hough_detector(frame, min_dist, p1, p2, min_radius, max_radius):
    """Return the bounding box for a basketball in an image, if present.

    Args:
        frame: Image to check for circles.
        min_dist: Minimum distance between detected circles
        p1: ?
        p2: ?
        min_radius: Min radius of detected circles.
        max_radius: Max radius of detected circles.

    Returns:
        Bounding box co-ordinates around the smallest detected circle.
        None if no circle can be found matching the input params. 

    """
    blur = cv2.GaussianBlur(frame, (9, 9), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(image=gray, method=cv2.HOUGH_GRADIENT,
                               dp=1, minDist=min_dist, param1=p1, param2=p2, minRadius=min_radius, maxRadius=max_radius)

    if circles is not None:
        # Circles are 'doublewrapped' in an extra list? Not sure what's going on, this seems to work.
        circles = circles[0]
        print(f"Circles found: {len(circles)}")
        # Taking the smallest circle here.
        (x, y, r) = min(circles, key=lambda x: math.pi * (x[2]**2))
        cv2.circle(frame, (x, y), r, (255, 0, 0), 2)
        # box_padding = 5 - tolerance around the ball
        width_height = (2*r) + 5
        return (x-r, y-r, width_height, width_height)
    else:
        print("No circles found.")
        return None


def morphological_transform(frame, opn_iter, cls_iter):
    """Apply morphological operators to a frame to reduce noise.

    Args:
        frame: Image to reduce noise in.
        opn_itr: Number of iterations of opening (erosion -> dilation) to apply.
        cls_iter: Number of iterations of closing (dilation -> erosion) to apply.

    Returns:
        The frame with noise reduced.

    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (19, 19), 0)

    block_size = 11
    thresh = cv2.adaptiveThreshold(
        gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block_size, 1)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(
        thresh, cv2.MORPH_OPEN, kernel, iterations=opn_iter)
    closing = cv2.morphologyEx(
        opening, cv2.MORPH_CLOSE, kernel, iterations=cls_iter)

    return closing


def basketball_candidate(contour, max_area, min_area, max_carea):
    """Apply filters to a contour to detect whether it's a basketball.
    Check for area of contours and area of circle enclosing the contour.

    Args:
        contour: Contour to check.
        max_area: If the area of the contour is larger than this, discard.
        min_area: If the area of the contour is smaller than this, discard.
        max_carea: If the area of the circle enclosing the contour is larger than this, discard.

    Returns:
        True if the contour is classified as a basketball, False if not.

    """

    # Check area of the contour
    area = cv2.contourArea(contour)
    if not min_area <= area <= max_area:
        return False

    # Check area of the min enclosing circle of the contour
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    circleArea = radius * radius * math.pi
    if circleArea > max_carea:
        return False

    # Color check?

    return True


def ball_detector(cap):
    """Take in a video and read frames until a ball which meets all the filters is detected.

    Args:
        cap: OpenCV video to read frames from.

    Returns:
        A bounding box around the detected basketball.

    """
    detected = False

    valid_contours = []
    frame_num = 0

    while True:
        ok, frame = cap.read()
        frame_num += 1

        if not ok:
            return None

        filtered_frame = morphological_transform(frame, 2, 3)

        contours, hierarchy = cv2.findContours(
            filtered_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print(
            f"Number of contours found in frame {frame_num}: {len(contours)}")

        for contour in contours:
            valid_contour = basketball_candidate(
                contour, max_area=500, min_area=300, max_carea=600)
            if valid_contour:
                return cv2.boundingRect(contour)

        box = hough_detector(frame, min_dist=200, p1=175,
                             p2=20, min_radius=10, max_radius=20)
        if box:
            print(f"Hough successful on {frame_num}")
            return box

    # cv2.drawContours(frame, valid_contours, -1, (255, 0, 0), 2)
    # show(frame)

    # return (0, 0, 0, 0)


def show(img):
    """Debug function to show opencv2 images."""
    cv2.imshow("Test", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    filepath = "./Data/FTVikas.mp4"
    cap = cv2.VideoCapture(filepath)
    bbox = ball_detector(cap)
    print(f"Bbox: {bbox}")
