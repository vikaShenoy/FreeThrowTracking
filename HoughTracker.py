import cv2
import numpy
import copy
import math


def ball_peak(shot_data):
    """Find the (x, y) co-ordinates for the ball at its peak in the trajectory."""
    (x, y) = min(shot_data, key=lambda x: x[1])
    return (int(x), int(y))


def ball_release(shot_data):
    """Find the (x, y) co-ordinates for the ball when its released by the player."""
    # X is a max, y is a min
    (x, y) = max(shot_data, key=lambda x: x[0] - x[1])
    return (int(x), int(y))


def ball_contact(shot_data, xpeak):
    """Find the (x, y) co-ordinates for the ball when it hits the backboard/hoop for the first time."""
    # Alg: want the co-ords where x is a min and y is a min. Find the minimum of the x * y.
    (xmin, ymax) = min(shot_data, key=lambda x: x[0] + x[1])
    height = int(ymax)

    result = ()
    xmax = -10000
    tolerance = 20

    for pos in shot_data:
        x, y = int(pos[0]), int(pos[1])
        if (height - tolerance < y < height + tolerance) and x > xmax and x < xpeak:
            xmax = x
            print(1)
            result = (x, y)

    return result


def detect_ball(frame):
    """Take in a frame from a video. Detect the basketball in the image and 
    return a bounding box around its location which can be used for tracking."""
    blur = cv2.GaussianBlur(frame, (9, 9), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(image=gray, method=cv2.HOUGH_GRADIENT,
                               dp=1, minDist=200, param1=150, param2=20, minRadius=6, maxRadius=20)

    # Copy the image for drawing on it
    if circles is not None:
        # Circles are 'doublewrapped' in an extra list? Not sure what's going on, this seems to work.
        circles = circles[0]
        # Taking the smallest circle here.
        (x, y, r) = min(circles, key=lambda x: math.pi * (x[2]**2))
        # Add some tolerance around the ball
        c = 5
        widthHeight = (2*r) + 5
        return (x-r, y-r, widthHeight, widthHeight)
    else:
        print("No circles found.")
        return


def find_launch_angle():
    """TODO"""


def track_ball():
    """Track a ball in a video of a basketball shot. Draw a box around the ball."""
    # TODO - handle the case where camera is on the other side of the shooter
    # CONSTRAINT: camera needs to be perpendicular to shot
    cap = cv2.VideoCapture("./FreeThrow.mp4")
    ok, initial_frame = cap.read()
    if not ok:
        print("Error reading inital frame")
        return

    bbox1 = detect_ball(initial_frame)

    tracker = cv2.TrackerCSRT_create()
    ok = tracker.init(initial_frame, bbox1)

    shot_data = []

    while cap.isOpened() and not (cv2.waitKey(1) & 0xFF == ord('q')):
        ok, frame = cap.read()
        if not ok:
            break
        timer = cv2.getTickCount()
        ok, bbox = tracker.update(frame)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        if ok:
            shot_data.append((bbox[0] + (bbox[2]/2), bbox[1] + (bbox[2]/2)))
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.imshow("Tracking", frame)

    # TESTING
    peak = ball_peak(shot_data)
    cv2.circle(initial_frame, ball_peak(shot_data), 20, (0, 0, 255), 1)
    cv2.circle(initial_frame, ball_release(shot_data), 20, (0, 0, 255), 1)
    cv2.circle(initial_frame, ball_contact(
        shot_data, peak[0]), 20, (0, 0, 255), 1)
    cv2.imshow("Ball positions", initial_frame)
    cv2.waitKey(0)


if __name__ == "__main__":
    # NOTE: main problem is that the hough circles require extremely finely tuned parameters.
    track_ball()
