import cv2
import numpy as np
import math

# How many contour points to find on the edge of a Hough circle
NUM_CIRCLE_POINTS = 50


def circle_to_contour(circle):
    """Convert a Hough circle to points representing the contour of the circle.

    Args:
        circle: (x, y, r) -> x, y co-ords of center and circle radius.

    Returns:
        Numpty array of the circle's contour points.
    """
    Cx, Cy, r = circle
    n = NUM_CIRCLE_POINTS
    total_angle = 2 * math.pi
    angle_step = total_angle / n
    contour = np.empty((n, 1, 2), dtype=np.int32)

    for i in range(n):
        angle = angle_step * i
        x = Cx + (r * math.sin(angle_step * i))
        y = Cy + (r * math.cos(angle_step * i))
        contour[i] = [x, y]

    return contour


def contour_to_box(contour):
    """Return the bounding rectangle of a contour (np array of points.)"""
    return cv2.boundingRect(contour)


def contour_to_circle(contour):
    """Return the min enclosing circle of a contour (np array of points.)"""
    return cv2.minEnclosingCircle(contour)


def circle_to_box(circle, padding):
    """Return the bounding box of a circle.

    Args:
        circle: (x, y, r) center and radius of a circle.
        padding: amount of padding to add for the box around the circle.

    Returns:
        Bounding box corners.

    """
    (x, y, r) = circle
    width_height = (2*r) + padding
    return (x-r, y-r, width_height, width_height)


def box_to_contour(bbox):
    """Return the 8 points of a bounding box."""
    contour = np.empty((8, 1, 2), dtype=np.int32)
    x1, y1, w, h = bbox
    contour[0] = [x1, y1]
    contour[1] = [x1 + (w/2), y1]
    contour[2] = [x1 + w, y1]
    contour[3] = [x1 + w, y1 + (h/2)]
    contour[4] = [x1 + w, y1 + h]
    contour[5] = [x1 + (w/2), y1 + h]
    contour[6] = [x1, y1 + h]
    contour[7] = [x1, y1 + (h/2)]

    return contour


def contour_centroid_color(frame, contour):
    """Find the HSV color value at the center of a contour.

    Args:
        frame: Image. 
        contour: list of point comprising the contour.

    Returns:
        [h, s, v] of the centroid of the contour.

    """
    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return [255, 255, 255]
    cX = int(moments["m10"] / moments["m00"])
    cY = int(moments["m01"] / moments["m00"])
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return hsv[cY, cX]


def bbox_center_color(frame, bbox):
    """."""
    x, y, w, h = bbox
    x1 = int(x + w)
    y1 = int(y + h)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return hsv[y1, x1]
