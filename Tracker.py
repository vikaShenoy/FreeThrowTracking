import cv2
import numpy
import copy
import math

from Util.BallDetection import detect_ball

# Radius for the circles to draw on the final image
RADIUS = 20
# Used to get the ball in the range of the release/peak/contact points
KEYPOINT_TOLERANCE = 10


def ball_peak(shot_data):
    """Find the (x, y) co-ordinates for the ball at its peak in the trajectory."""
    return min(shot_data, key=lambda x: x[1])


def ball_release(shot_data, xpeak):
    """Find the (x, y) co-ordinates for the ball when its released by the player."""
    (x1, y1) = max(shot_data, key=lambda x: x[0])
    ymax = math.inf

    result = None
    for (x, y) in shot_data:
        if (x1 - KEYPOINT_TOLERANCE <= x <= x1 + KEYPOINT_TOLERANCE) and y < ymax and x > xpeak:
            ymax = y
            result = (x, y)
    return result


def ball_contact(shot_data, xpeak):
    """Find the (x, y) co-ordinates for the ball when it hits the backboard/hoop for the first time."""
    (x1, y1) = min(shot_data, key=lambda x: x[0] + x[1])
    xmax = -math.inf

    result = None
    for (x, y) in shot_data:
        if (y1 - KEYPOINT_TOLERANCE < y < y1 + KEYPOINT_TOLERANCE) and x > xmax and x < xpeak:
            xmax = x
            result = (x, y)
    return result


def calculate_launch_angle(shot_data, p1, frame, n=3):
    """Take in shot data and return the angle at which the ball was released by the shooter."""
    p2 = ()
    # Find a frame slightly after the release frame to calculate the launch angle with
    for i, (x, y) in enumerate(shot_data):
        if (x, y) == p1:
            p2 = shot_data[i + n]

    xdiff = p1[0] - p2[0]
    ydiff = p1[1] - p2[1]

    print(f"P1: {p1}")
    print(f"p2: {p2}")
    print(f"Xdiff: {xdiff}")
    print(f"Ydiff: {ydiff}")

    return math.degrees(math.atan(ydiff / xdiff))


def calculate_throw_time(shot_data, release, contact, frame_rate):
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


def calculate_launch_velocity(throw_time, release_angle):
    """."""
    c = 4.9
    return abs(c * (throw_time / math.sin(release_angle)))


def track_ball(video_path):
    """Track a ball in a video of a basketball shot. Draw a box around the ball."""
    # TODO - handle the case where camera is on the other side of the shooter
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    shot_data = []

    bbox = detect_ball(cap)
    # bbox = cv2.selectROI("Select", initial_frame)
    if not bbox:
        print("Error detecting ball")
        return

    ok, initial_frame = cap.read()
    if not ok:
        print("Error reading inital frame")
        return

    tracker = cv2.TrackerCSRT_create()
    ok = tracker.init(initial_frame, bbox)

    while cap.isOpened() and not (cv2.waitKey(1) & 0xFF == ord('q')):
        ok, frame = cap.read()
        if not ok:
            break
        timer = cv2.getTickCount()
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        ok, bbox = tracker.update(frame)
        if ok:
            shot_data.append(
                (int(bbox[0] + (bbox[2]/2)), int(bbox[1] + (bbox[2]/2))))
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)

        cv2.imshow("Tracking", frame)

    # Extract keypoints
    peak = ball_peak(shot_data)
    release = ball_release(shot_data, peak[0])
    contact = ball_contact(shot_data, peak[0])

    # Calculate the launch angle
    angle = calculate_launch_angle(
        shot_data=shot_data, p1=release, frame=initial_frame, n=10)
    print(f"Angle: {angle}")

    # Calculate the throwing velocity
    throw_time = calculate_throw_time(shot_data, release, contact, frame_rate)
    velocity = calculate_launch_velocity(
        throw_time=throw_time, release_angle=angle)
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
    track_ball(video_path="./Data/FTVikas.mp4")
