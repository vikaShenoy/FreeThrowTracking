import cv2
import numpy as np
import math

# How many contour points to find on the edge of a Hough circle
NUM_CIRCLE_POINTS = 30

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
    """."""
    return cv2.boundingRect(contour)
