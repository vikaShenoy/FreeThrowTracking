"""Detect a basketball in an image. First step in ball tracking."""

import cv2
import numpy as np
import math

from .Util import circle_to_contour, contour_to_box

# How much tolerance to add around the detected basketball
BOX_PADDING = 1

# Hough parameter constants
CANNY_THRESH = 175
ACCUM_THRESH = 20
MIN_RADIUS = 10
MAX_RADIUS = 19
MIN_DIST = 200

# Valid contour constants
MAX_AREA = math.pi * (MAX_RADIUS ** 2)
MIN_AREA = math.pi * (MIN_RADIUS ** 2)
MAX_CAREA = MAX_AREA + 100

MIN_VALUE = 0
MAX_VALUE = 150


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


def hough_detector(frame, min_dist, canny_thresh, accum_thresh, min_radius, max_radius):
    """Return the bounding box for a basketball in an image, if present.

    Args:
        frame: Image to check for circles.
        min_dist: Minimum distance between detected circles
        canny_thresh: Internal threshold for the canny edge detector. 
        Higher of the two thresholds passed to the canny detector. 
        accum_thresh: threshold for center detection. Smaller values lead to more
        false circles being detected.
        min_radius: Min radius of detected circles.
        max_radius: Max radius of detected circles.

    Returns:
        Bounding box co-ordinates around the smallest detected circle.
        None if no circle can be found matching the input params. 

    """
    blur = cv2.GaussianBlur(frame, (9, 9), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(image=gray, method=cv2.HOUGH_GRADIENT,
                               dp=1, minDist=min_dist, param1=canny_thresh,
                               param2=accum_thresh, minRadius=min_radius, maxRadius=max_radius)

    if circles is not None:
        # Circles are 'doublewrapped' in an extra list? Not sure what's going on, this seems to work.
        circles = circles[0]
        print(f"Circles found: {len(circles)}")

        for circle in circles:
            contour = circle_to_contour(circle)
            valid = valid_contour(
                frame, contour, MAX_AREA, MIN_AREA, MAX_CAREA)
            if valid:
                return contour_to_box(contour)

    print("No valid circles found.")
    return None


def valid_contour(frame, contour, max_area, min_area, max_carea):
    """Apply filters to a contour to detect whether it's a basketball.
    Check for area of contours and area of circle enclosing the contour.

    Args:
        frame: Image the contour comes from. Used for color checking.
        contour: Contour to check.
        max_area: If the area of the contour is larger than this, discard.
        min_area: If the area of the contour is smaller than this, discard.
        max_carea: If the area of the circle enclosing the contour is larger than this, discard.

    Returns:
        Box around the contour if the contour is classified as a basketball.
        None if the contour is not classified as a basketball.

    """

    # Check area of the contour
    area = cv2.contourArea(contour)
    if not min_area <= area <= max_area:
        print("Contour failed area check")
        return False

    # Check area of the min enclosing circle of the contour
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    circle_area = radius * radius * math.pi
    if circle_area > max_carea:
        print("Contour failed circle area check")
        return False

    # Color check
    moments = cv2.moments(contour)
    cX = int(moments["m10"] / moments["m00"])
    cY = int(moments["m01"] / moments["m00"])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    [h, s, v] = hsv[cY, cX]
    if not MIN_VALUE <= v <= MAX_VALUE:
        print("Contour failed color value check")
        return False

    return True


def detect_ball(cap):
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

        # cv2.drawContours(frame, contours, -1, (255, 0, 0), 2)
        # show(frame)

        print(
            f"Number of contours found in frame {frame_num}: {len(contours)}")

        # for contour in contours:
        #     valid = valid_contour(
        #         frame, contour, MAX_AREA, MIN_AREA, MAX_CAREA)
        #     if valid:
        #         return contour_to_box(contour)

        bbox = hough_detector(frame, MIN_DIST, CANNY_THRESH,
                              ACCUM_THRESH, MIN_RADIUS, MAX_RADIUS)
        if bbox:
            print(f"Hough successful on {frame_num}")
            return bbox

    # cv2.drawContours(frame, valid_contours, -1, (255, 0, 0), 2)
    # show(frame)
    # return (0, 0, 0, 0)


def show(img):
    """Debug function to show opencv2 images."""
    cv2.imshow("Test", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    filepath = "./Data/FT.mp4"
    cap = cv2.VideoCapture(filepath)
    bbox = detect_ball(cap)
    print(f"Bbox: {bbox}")
