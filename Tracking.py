
import cv2
import numpy as np
import copy


def color_segmentation(img, lower, upper):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask=mask)
    return result


def hough_circles(img, segmentedImage):
    # Try on the plain image first
    blur = cv2.GaussianBlur(segmentedImage, (9, 9), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray", gray)
    circles = cv2.HoughCircles(image=gray, method=cv2.HOUGH_GRADIENT,
                               dp=1, minDist=200, param1=150, param2=20, minRadius=6, maxRadius=20)

    result = img.copy()
    if circles is not None:
        print(1)
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(result, (i[0], i[1]), i[2], (0, 255, 0), 2)
    else:
        print(2)
    cv2.imshow("Circles", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    # Use a combination of masking and frame differencing
    # Get the 'orange' pixels from the first three frames
    # Subtract the second segmented frame from the first
    # Random: Use brightness to improve video quality?

    # Setup - get the first frame from the video.
    cap = cv2.VideoCapture("./FreeThrow.mp4")
    num = 100
    cap.set(cv2.CAP_PROP_POS_FRAMES, num - 1)
    ok, frame = cap.read()
    if not ok:
        print("Error reading first frame")

    # 1) Use the color segmentation to get the regions which are in color range

    lower = np.array([-68, 62, 37])
    upper = np.array([72, 202, 117])
    segmentedImage = color_segmentation(frame, lower, upper)
    # DEBUG
    cv2.imshow("Mask", segmentedImage)

    # 2) Use the Hough Circle Transform to find the circles in that mask
    hough_circles(frame, segmentedImage)

    # 3) Draw the bounding box around the correct circle -> THE GOAL IS TO GET OUR INITAL BBOX
    # 4) Use KCF to track
    # 5) Collect metrics during this tracking process (during number 4 probably)
    # 6) Overlay the tracking with stats and print them at the end

    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
