"""Detect a basketball in an image. First step in ball tracking."""

import cv2
import numpy as np
import math

import Utility.Util as Util
# import Util as Util

# Hough parameter constants
CANNY_THRESH = 100
ACCUM_THRESH = 18
MIN_CANNY_RADIUS = 10
MAX_CANNY_RADIUS = 25
MIN_DIST = 110
DP = 1

# Valid contour constants
MIN_CONTOUR_AREA = 400
MAX_CONTOUR_AREA = 1200
MIN_CIRCLE_RADIUS = 9
MAX_CIRCLE_RADIUS = 20
MIN_CIRCLE_AREA = (math.pi * (MIN_CIRCLE_RADIUS ** 2))
MAX_CIRCLE_AREA = (math.pi * (MAX_CIRCLE_RADIUS ** 2))

# HSV ranges
# MIN_HSV = [0, 60, 60]
MIN_HSV = [0, 130, 40]
MAX_HSV = [180, 230, 120]

BOX_PADDING = 15


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
        for circle in circles:
            contour = Util.circle_to_contour(circle)
            valid = valid_contour(frame, contour)
            if valid:
                cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)
                show(frame)
                return Util.circle_to_box(circle, padding=BOX_PADDING)

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

    # cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
    # show(frame)

    for contour in contours:
        valid = valid_contour(frame, contour)
        if valid:
            cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)
            show(frame)
            (x, y), r = Util.contour_to_circle(contour)
            return Util.circle_to_box((x, y, r), BOX_PADDING)

    return None


def valid_hsv(hsv):
    """Return True if an array of hsv values falls within valid ranges.
    Valid ranges are global constants."""
    [h, s, v] = hsv
    if not MIN_HSV[0] <= h <= MAX_HSV[0]:
        return False
    if not MIN_HSV[1] <= s <= MAX_HSV[1]:
        return False
    if not MIN_HSV[2] <= v <= MAX_HSV[2]:
        return False
    return True


def valid_tracker_box(frame, bbox):
    """Return True if the bbox around a basketball is accurate.
    Checks the color of the centroid for an hsv range."""
    contour = Util.box_to_contour(bbox)

    [h, s, v] = Util.contour_centroid_color(frame, contour)
    min_hsv = [0, 0, 40]
    max_hsv = [180, 255, 140]
    if not min_hsv[0] <= h <= max_hsv[0]:
        return False
    if not min_hsv[1] <= s <= max_hsv[1]:
        return False
    if not min_hsv[2] <= v <= max_hsv[2]:
        return False
    return True


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
    hsv = Util.contour_centroid_color(frame, contour)
    if not valid_hsv(hsv):
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

    # cap = skip(cap, 10)

    while True:
        ok, frame = cap.read()
        frame_num += 1
        if not ok:
            return 0, None

        bbox = hough_detector(frame)
        if bbox:
            print(f"Hough successful on frame {frame_num}")
            return frame_num, bbox

        bbox = contour_detector(frame)
        if bbox:
            print(f"Contour successful on frame {frame_num}")
            return frame_num, bbox

        print(f"No ball detected on frame: {frame_num}")


def show(img):
    """Debug function to show opencv2 images."""
    cv2.imshow("Test", img)
    cv2.waitKey(0)


def skip(cap, n):
    """Debug function to skip a video ahead n frames."""
    for i in range(n):
        ok, frame = cap.read()
    return cap


if __name__ == "__main__":
    filepath = "./Data/FTVikas3.mp4"
    cap = cv2.VideoCapture(filepath)
    bbox = detect_ball(cap)
    print(f"Bbox: {bbox}")
