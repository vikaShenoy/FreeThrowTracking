"""Detect a basketball in an image. First step in ball tracking."""

import cv2
import numpy as np
import math

from .Util import circle_to_contour, contour_to_box, circle_to_box, contour_to_circle

# Hough parameter constants
CANNY_THRESH = 120
ACCUM_THRESH = 14
MIN_CANNY_RADIUS = 10
MAX_CANNY_RADIUS = 20
MIN_DIST = 120
DP = 1

# Valid contour constants
MIN_CONTOUR_AREA = 400
MAX_CONTOUR_AREA = 1000
MIN_CIRCLE_RADIUS = 10
MAX_CIRCLE_RADIUS = 18
MIN_CIRCLE_AREA = (math.pi * (MIN_CIRCLE_RADIUS ** 2))
MAX_CIRCLE_AREA = (math.pi * (MAX_CIRCLE_RADIUS ** 2))

# HSV ranges
MIN_HSV = [0, 140, 60]
MAX_HSV = [180, 230, 120]

BOX_PADDING = 5


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


def hough_detector(frame):
    """Use the Hough Circles algorithm to detect a basketball, if present.
    Convert detected circles to contours and filter based on 
    area and color to check for validity.

    Args:
        frame: Image to check for circles.

    Returns:
        Bounding box co-ordinates around a valid basketball.
        None if no circle can be found matching the input params.

    """
    blur = cv2.GaussianBlur(frame, (9, 9), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(image=gray, method=cv2.HOUGH_GRADIENT,
                               dp=1, minDist=MIN_DIST, param1=CANNY_THRESH,
                               param2=ACCUM_THRESH, minRadius=MIN_CANNY_RADIUS, maxRadius=MAX_CANNY_RADIUS)

    if circles is not None:
        # Circles are 'doublewrapped' in an extra list? Not sure what's going on, this seems to work.
        circles = circles[0]
        print(f"Circles found: {len(circles)}")

        for circle in circles:
            contour = circle_to_contour(circle)
            valid = valid_contour(frame, contour)
            if valid:
                cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)
                show(frame)
                return circle_to_box(circle, padding=BOX_PADDING)

    return None


def contour_detector(frame):
    """Find all contours in an image and check if any of them match
    the features of a basketball - using area and color. 

    Args:
        frame: Image to find contours in.

    Returns:
        A bounding box around the valid basketball contour, if found.
        None if no basketball could be detected. 

    """

    filtered_frame = morphological_transform(frame, 2, 3)
    contours, hierarchy = cv2.findContours(
        filtered_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Contours found: {len(contours)}")
    for contour in contours:
        valid = valid_contour(frame, contour)
        if valid:
            cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)
            show(frame)
            (x, y), r = contour_to_circle(contour)
            return circle_to_box((x, y, r), BOX_PADDING)

    return None


def valid_contour(frame, contour):
    """Apply filters to a contour to detect whether it's a basketball.
    Check for area of contour, area of circle enclosing the contour, and
    hsv color of the contour's centroid.

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
    contour_area = cv2.contourArea(contour)
    if not MIN_CONTOUR_AREA <= contour_area <= MAX_CONTOUR_AREA:
        return False

    # Check area of the circle
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    circle_area = radius * radius * math.pi
    if not MIN_CIRCLE_AREA <= circle_area <= MAX_CIRCLE_AREA:
        return False

    # Check color of the centroid of the contour
    moments = cv2.moments(contour)
    cX = int(moments["m10"] / moments["m00"])
    cY = int(moments["m01"] / moments["m00"])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    [h, s, v] = hsv[cY, cX]

    if not MIN_HSV[0] <= h <= MAX_HSV[0]:
        return False
    if not MIN_HSV[1] <= s <= MAX_HSV[1]:
        return False
    if not MIN_HSV[2] <= v <= MAX_HSV[2]:
        return False

    return True


def detect_ball(cap):
    """Take in a video and read frames until a valid basketball is detected.

    Args:
        cap: OpenCV video to read frames from.

    Returns:
        A bounding box around the detected basketball.
        None if the basketball couldn't be detected.

    """
    detected = False

    valid_contours = []
    frame_num = 0

    while True:
        ok, frame = cap.read()
        frame_num += 1
        if not ok:
            return None

        bbox = hough_detector(frame)
        if bbox:
            print(f"Hough successful on {frame_num}")
            return bbox

        bbox = contour_detector(frame)
        if bbox:
            print(f"Contour successful on {frame_num}")
            return bbox

        print(f"No ball detected on frame: {frame_num}")


def show(img):
    """Debug function to show opencv2 images."""
    cv2.imshow("Test", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    filepath = "./Data/FT.mp4"
    cap = cv2.VideoCapture(filepath)
    bbox = detect_ball(cap)
    print(f"Bbox: {bbox}")
