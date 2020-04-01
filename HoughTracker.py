import cv2
import numpy
import copy
import math


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
        (x, y, r) = min(circles, key=lambda x: math.pi * (x[2]**2))
        # Add some tolerance around the ball
        c = 5
        widthHeight = (2*r) + 5
        return (x-r, y-r, widthHeight, widthHeight)
    else:
        print("No circles found.")
        return


def track_ball():
    """Track a ball in a video of a basketball shot. Draw a box around the ball."""
    cap = cv2.VideoCapture("./FreeThrow.mp4")
    ok, frame = cap.read()
    if not ok:
        print("Error reading inital frame")
        return

    bbox1 = detect_ball(frame)

    tracker = cv2.TrackerCSRT_create()
    ok = tracker.init(frame, bbox1)

    while cap.isOpened() and not (cv2.waitKey(1) & 0xFF == ord('q')):
        ok, frame = cap.read()
        if not ok:
            break
        timer = cv2.getTickCount()
        ok, bbox = tracker.update(frame)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        if ok:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

        cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)
        cv2.imshow("Tracking", frame)


if __name__ == "__main__":
    # NOTE: main problem is that the hough circles require extremely finely tuned parameters.
    track_ball()
