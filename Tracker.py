"""
Author: Vikas Shenoy
Program to calculate the launch angle and velocity from side-on free throw video
using OpenCV 2 and computer vision techniques. This file handles the main tracking
loop as well as parameter calculation, and displays tracking/results to the user. 
"""


import cv2
import numpy as np
import copy
import math

from Utility.BallDetection import detect_ball, show, valid_tracker_box

# Radius for the circles to draw on the final image
RADIUS = 20
# Used to get the ball in the range of the release/peak/contact points
KEYPOINT_TOLERANCE = 5
# How many consecutive frames need to fail the detector check for tracking to fail
INACCURACY_TOLERANCE = 10
# Constants for shot release side identification
RIGHT = 0
LEFT = 1

GOLD = (0, 204, 255)
RED = (0, 0, 255)


def ball_peak(shot_data):
    """Find the (x, y) co-ordinates for the ball at its peak in the trajectory.
    NOTE: the y-axis is in the negative direction. So the lowest y-value
    indicates the ball at its highest point.

    Args:
        shot_data: list of (x, y) locations of the ball during the shot.

    Returns:
        The index of the shot location in shot_data which contains the ball peak.
    """
    peak_index = None

    ymin = math.inf
    for index, (x, y) in enumerate(shot_data):
        if y < ymin:
            ymin = y
            peak_index = index

    return peak_index


def release_side(shot_data, peak_index):
    """Find the side the basketball is released on.
    Useful for further calculations on angles.

    Args:
        shot_data: List of shot_coordinates for the ball.
        peak_index: Index of the ball at its highest point. We assume
        this is on its shot path.

    Returns:
        0 if the ball is shot from right to left.
        1 if the ball is shot from left to right.

    """
    c = 2
    x_prev = shot_data[peak_index - c][0]
    x_post = shot_data[peak_index + c][0]
    x_peak = shot_data[peak_index][0]

    if x_prev <= x_peak <= x_post:
        return LEFT
    return RIGHT


def ball_release_right(shot_data, xpeak):
    """Find the (x, y) co-ordinates for the ball when its released by the player on the right side.

    Args:
        shot_data: list of (x, y) positions of the ball from tracking during the shot.
        xpeak: the x coordiante for the ball at its peak position

    Returns:
        (x, y) location of where the ball was released from on the right side of the video.

    """
    (x1, y1) = max(shot_data, key=lambda x: x[0])
    ymax = math.inf

    result = None
    for (x, y) in shot_data:
        if (x1 - KEYPOINT_TOLERANCE <= x <= x1 + KEYPOINT_TOLERANCE) and y < ymax and x > xpeak:
            ymax = y
            result = (x, y)
    return result


def ball_release_left(shot_data, xpeak):
    """Find the (x, y) co-ordinates for the ball when its released by the player on the left side.

    Args:
        shot_data: list of (x, y) positions of the ball from tracking during the shot.
        xpeak: the x coordiante for the ball at its peak position

    Returns:
        (x, y) location of where the ball was released from on the left side of the video.

    """
    (x1, y1) = min(shot_data, key=lambda x: x[0])
    ymax = math.inf

    result = None
    for (x, y) in shot_data:
        if (x1 - KEYPOINT_TOLERANCE <= x <= x1 + KEYPOINT_TOLERANCE) and y < ymax and x < xpeak:
            ymax = y
            result = (x, y)
    return result


def calculate_launch_angle(shot_data, p1, n=3):
    """Take in shot data and return the angle at which the ball was released by the shooter.

    Args:
        shot_data: List of (x, y) co-ordinates from the center of the ball tracker.
        p1: First point to calculate angle from - release point.
        n: How many frames after the inital frame to calculate angle with.
        Large values will form a large triangle.

    Returns:
        The launch angle in degrees.

    """
    p2 = ()
    # Find a frame slightly after the release frame to calculate the launch angle with
    for i, (x, y) in enumerate(shot_data):
        if (x, y) == p1:
            p2 = shot_data[i + n]

    xdiff = abs(p1[0] - p2[0])
    ydiff = p1[1] - p2[1]
    return math.degrees(math.atan(ydiff / xdiff))


def calculate_launch_velocity(throw_time, release_angle):
    """Find the velocity the ball is released with.

    Args:
        throw_time: Time for the ball to go from release to peak locations.
        release_angle: angle in degreees the ball was released with. 

    Returns:
        (float) velocity of the ball at launch, in m/s.

    """
    gravity_const = 9.8
    return abs(gravity_const * (throw_time / np.sin(np.deg2rad(release_angle))))


def calculate_throw_time(shot_data, release, peak, frame_rate):
    """Find how long it takes the ball to go from release to contact.

    Args:
        shot_data: List of (x, y) co-ordinates from the center of the ball tracker.
        release: (x, y) location of ball release.
        contact: (x, y) location of ball contact.
        frame_rate: video frame rate.

    Returns:
        Throw time in seconds.
    """
    p1 = 0
    p2 = 0

    for i, pos in enumerate(shot_data):
        if pos == release:
            p1 = i
        elif pos == peak:
            p2 = i

    frame_diff = p2 - p1
    return frame_diff / frame_rate


def extract_keypoints(shot_data):
    """Find the peak and release x,y locations from shot data.
    Args:
        shot_data: List of (x, y) locations from ball tracking.

    Returns:
        release, peak: (x,y), (x,y) coordinates of keypoints.
    """
    peak_index = ball_peak(shot_data)
    peak = shot_data[peak_index]

    side = release_side(shot_data, peak_index)
    if side == RIGHT:
        release = ball_release_right(shot_data, peak[0])
    elif side == LEFT:
        release = ball_release_left(shot_data, peak[0])

    return release, peak


def display_stats(frame, keypoints, angle, velocity):
    """Display the ball release/peak/contact.
    Display the launch velocity and release angle."""
    cv2.circle(frame, keypoints[0], RADIUS, RED, 2)
    cv2.circle(frame, keypoints[1], RADIUS, RED, 2)
    cv2.putText(frame, f"Launch angle: {round(angle, 1)}deg",
                (550, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.75, GOLD, 2)
    cv2.putText(frame, f"Velocity: {round(velocity, 1)}m/s",
                (550, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.75, GOLD, 2)
    cv2.imshow("Ball positions", frame)
    cv2.waitKey(0)


def track_ball(cap, initial_frame, tracker, bbox):
    """Track a basketball during a shot.

    Args:
        cap: OpenCV video capture object to read frames from.
        initial_frame: Starting frame to use for initialising tracker.
        tracker: Object tracker.
        bbox: Co-ordinates for the bounding box of the ball to track.

    Returns:
        A list of shot_data (x, y) locations of the center of the
        bounding box, representing the basketball's shot locations.

    """
    shot_data = []
    inaccuracy_count = 0

    ok = tracker.init(initial_frame, bbox)
    if not ok:
        print("Error initialising tracker")
        return shot_data

    while cap.isOpened() and not (cv2.waitKey(1) & 0xFF == ord('q')):
        ok, frame = cap.read()
        if not ok:
            break

        ok, bbox = tracker.update(frame)
        if not valid_tracker_box(frame, bbox):
            inaccuracy_count += 1
        else:
            inaccuracy_count = 0

        if inaccuracy_count >= INACCURACY_TOLERANCE:
            print(
                f"Tracking failure detected on frame {cap.get(cv2.CAP_PROP_POS_FRAMES)}")
            shot_data = shot_data[:-INACCURACY_TOLERANCE]
            current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

            redetect_frames, bbox = detect_ball(cap)
            if not bbox:
                print("Ball could not be re-detected.")
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES,
                    current_frame + redetect_frames - 1)
            ok, frame = cap.read()
            tracker = cv2.TrackerCSRT_create()
            tracker.init(frame, bbox)
            inaccuracy_count = 0

        shot_data.append(
            (int(bbox[0] + (bbox[2]/2)), int(bbox[1] + (bbox[2]/2))))
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2, 1)
        cv2.imshow("Tracking", frame)
    return shot_data


def main(video_path):
    """Detect and track a basketball during a freethrow.
    Find the release, peak and contact points of the shot.
    Calculate the launch angle and velocity of the shot.

    Args:
        video_path: Path to video file of shot to load.

    """
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    n_detection_frames, bbox = detect_ball(cap)
    # Can be used to manually detect the ball
    # bbox = cv2.selectROI("Select", initial_frame)
    if not bbox:
        print("Error detecting ball")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, n_detection_frames - 1)
    ok, initial_frame = cap.read()
    if not ok:
        print("Error reading inital frame")
        return shot_data

    tracker = cv2.TrackerCSRT_create()
    shot_data = track_ball(cap, initial_frame, tracker, bbox)
    keypoints = extract_keypoints(shot_data)
    (release, peak) = keypoints

    try:
        angle = calculate_launch_angle(shot_data, release, n=10)
        throw_time = calculate_throw_time(
            shot_data, release, peak, frame_rate)
        velocity = calculate_launch_velocity(throw_time, angle)
        display_stats(initial_frame, keypoints, angle, velocity)

    except Exception:
        raise
        exit


if __name__ == "__main__":
    main(video_path="./Data/FTVikas5.mp4")
