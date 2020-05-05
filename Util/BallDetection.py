"""Detect a basketball in an image. First step in ball tracking."""

import cv2
import numpy as np
import math

from .Util import circle_to_contour, contour_to_box, morphological_transform

# How much tolerance to add around the detected basketball
BOX_PADDING = 1


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
                contour, max_area=600, min_area=300, max_carea=700)
            if valid:
                return contour_to_box(contour)

        # (x, y, r) = min(circles, key=lambda x: math.pi * (x[2]**2))
        # # cv2.circle(frame, (x, y), r, (255, 0, 0), 2)
        # width_height = (2*r) + BOX_PADDING
        # return (x-r, y-r, width_height, width_height)
    print("No valid circles found.")
    return None


def valid_contour(contour, max_area, min_area, max_carea):
    """Apply filters to a contour to detect whether it's a basketball.
    Check for area of contours and area of circle enclosing the contour.

    Args:
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
        return False

    # Check area of the min enclosing circle of the contour
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    circleArea = radius * radius * math.pi
    if circleArea > max_carea:
        return False

    # Color check?

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

    while not detected:
        ok, frame = cap.read()
        frame_num += 1

        if not ok:
            return None

        filtered_frame = morphological_transform(frame, 2, 3)

        contours, hierarchy = cv2.findContours(
            filtered_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        print(
            f"Number of contours found in frame {frame_num}: {len(contours)}")

        # for contour in contours:
        #     valid = valid_contour(
        #         contour, max_area=500, min_area=300, max_carea=600)
        #     if valid:
        #         valid_contours.append(contour)
        #         print(f"Contour detector successful on: {frame_num}")
        #         detected = True
        #         # return bbox

        bbox = hough_detector(frame, min_dist=200, canny_thresh=175,
                              accum_thresh=20, min_radius=10, max_radius=20)
        if bbox:
            print(f"Hough successful on {frame_num}")
            return bbox

    # cv2.drawContours(frame, valid_contours, -1, (255, 0, 0), 2)
    show(frame)

    return (0, 0, 0, 0)


def show(img):
    """Debug function to show opencv2 images."""
    cv2.imshow("Test", img)
    cv2.waitKey(0)


if __name__ == "__main__":
    filepath = "./Data/FT.mp4"
    cap = cv2.VideoCapture(filepath)
    bbox = detect_ball(cap)
    print(f"Bbox: {bbox}")
