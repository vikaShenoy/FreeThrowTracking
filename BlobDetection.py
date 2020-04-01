import cv2
import numpy


def blob_detector(params):
    cap = cv2.VideoCapture("./FreeThrow.mp4")
    ok, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(gray)
    detected = cv2.drawKeypoints(frame, keypoints, None, (0, 0, 255),
                                 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imshow("detected", detected)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    params = cv2.SimpleBlobDetector_Params()

    # Set thresholds for the image binarization.
    params.minThreshold = 10
    params.maxThreshold = 200
    params.thresholdStep = 10

    # Filter by colour.
    params.filterByColor = False
    params.blobColor = 10

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 150
    params.maxArea = 300

    # Filter by Circularity
    params.filterByCircularity = False
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.10

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.8

    blob_detector(params)
