import cv2
import numpy as np
import math

# How many contour points to find on the edge of a Hough circle
NUM_CIRCLE_POINTS = 30


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
