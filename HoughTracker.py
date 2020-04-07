import cv2
import numpy
import copy
import math

BOX_PADDING = 0
RADIUS = 20


def ball_peak(shot_data):
    """Find the (x, y) co-ordinates for the ball at its peak in the trajectory."""
    return min(shot_data, key=lambda x: x[1])


def ball_release(shot_data):
    """Find the (x, y) co-ordinates for the ball when its released by the player."""
    # X is a max, y is a min
    return max(shot_data, key=lambda x: x[0] - x[1])


def ball_contact(shot_data, xpeak):
    """Find the (x, y) co-ordinates for the ball when it hits the backboard/hoop for the first time."""
    # Alg: want the co-ords where x is a min and y is a min. Find the minimum of the x * y.
    (xmin, ymax) = min(shot_data, key=lambda x: x[0] + x[1])
    height = ymax

    result = ()
    xmax = -10000
    tolerance = 20

    for (x, y) in shot_data:
        if (height - tolerance < y < height + tolerance) and x > xmax and x < xpeak:
            xmax = x
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
        RADIUS = r
        # Add some tolerance around the ball
        widthHeight = (2*r) + BOX_PADDING
        return (x-r, y-r, widthHeight, widthHeight)
    else:
        print("No circles found.")
        return


def find_launch_angle(shot_data, p1, frame, n=3):
    """Take in shot data and return the angle at which the ball was released by the shooter."""
    # TODO - implement with a more sophisticated approach (if one exists)

    p2 = ()
    # Find a frame slightly after the release frame to calculate the launch angle with
    for i, (x, y) in enumerate(shot_data):
        if (x, y) == p1:
            p2 = shot_data[i + n]

    xdiff = p1[0] - p2[0]
    ydiff = p1[1] - p2[1]

    # Testing
    # cv2.circle(frame, p1, RADIUS, (255, 0, 0), 2)
    # cv2.circle(frame, p2, RADIUS, (255, 0, 0), 2)
    # cv2.imshow("Angle finding", frame)
    # cv2.waitKey(0)
    return math.degrees(math.atan(ydiff / xdiff))


def find_throw_time(shot_data, release, contact, frame_rate):
    """Find how long it takes the ball to go from release to contact."""
    p1 = 0
    p2 = 0

    for i, pos in enumerate(shot_data):
        if pos == release:
            p1 = i
        elif pos == contact:
            p2 = i

    frame_diff = p2 - p1
    return frame_diff / frame_rate


def find_launch_velocity(throw_time, release_angle):
    """."""
    c = 4.9
    return c * (throw_time / math.sin(release_angle))


def track_ball():
    """Track a ball in a video of a basketball shot. Draw a box around the ball."""
    # TODO - handle the case where camera is on the other side of the shooter
    # CONSTRAINT: camera needs to be perpendicular to shot
    cap = cv2.VideoCapture("./FreeThrow.mp4")
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
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
            shot_data.append(
                (int(bbox[0] + (bbox[2]/2)), int(bbox[1] + (bbox[2]/2))))
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

        cv2.imshow("Tracking", frame)

    # Extract keypoints
    peak = ball_peak(shot_data)
    release = ball_release(shot_data)
    contact = ball_contact(shot_data, peak[0])

    # Calculate the launch angle
    angle = find_launch_angle(
        shot_data=shot_data, p1=release, frame=initial_frame, n=1)
    print(f"Angle: {angle}")

    # Calculate the throwing velocity
    throw_time = find_throw_time(shot_data, release, contact, frame_rate)
    velocity = find_launch_velocity(throw_time=throw_time, release_angle=angle)
    print(f"Velocity: {velocity}")

    # TESTING to see positions
    cv2.circle(initial_frame, peak, RADIUS, (0, 0, 255), 2)
    cv2.circle(initial_frame, release, RADIUS, (0, 0, 255), 2)
    cv2.circle(initial_frame, contact, RADIUS, (0, 0, 255), 2)
    cv2.putText(initial_frame, f"Launch angle: {round(angle, 1)}deg",
                (200, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
    cv2.putText(initial_frame, f"Velocity: {round(velocity, 1)}m/s",
                (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
    cv2.imshow("Ball positions", initial_frame)
    cv2.waitKey(0)


if __name__ == "__main__":
    # NOTE: main problem is that the hough circles require extremely finely tuned parameters.
    track_ball()
