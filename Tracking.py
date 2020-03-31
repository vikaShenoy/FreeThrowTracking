
import cv2
import numpy as np
import copy

def color_segmentation(img, lower, upper):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  return cv2.inRange(hsv, lower, upper)

def hough_circles(img, mask):
  # Try on the plain image first
  blur = cv2.GaussianBlur(img, (9,9), 0)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  circles = cv2.HoughCircles(image=mask, method=cv2.HOUGH_GRADIENT, dp=1, minDist=200, param1=300, param2=20, minRadius=10, maxRadius=30)

  result = img.copy()
  if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0,:]:
      cv2.circle(result, (i[0], i[1]), i[2], (0,255,0), 2)
  cv2.imshow("Circles", result)

  cv2.waitKey(0)
  cv2.destroyAllWindows()


def main():
  # Setup - get the first frame from the video.
  cap = cv2.VideoCapture("./FreeThrow.mp4")
  ok, frame = cap.read()
  if not ok:
    print("Error reading first frame")

  # 1) Use the color segmentation to get the regions which are in color range
  lower = np.array([-68, 59, 39])
  upper = np.array([72, 199, 119])
  mask = color_segmentation(frame, lower, upper)

  # DEBUG
  cv2.imshow("Mask", mask)

  # 2) Use the Hough Circle Transform to find the circles in that mask
  hough_circles(frame, mask)

  # 3) Draw the bounding box around the correct circle -> THE GOAL IS TO GET OUR INITAL BBOX
  # 4) Use KCF to track

  cv2.waitKey(0)
  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()