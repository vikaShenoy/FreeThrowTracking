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
