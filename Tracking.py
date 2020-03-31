
import cv2
import numpy as np

def color_segmentation():
  """Find the orange objects in the first frame of a video."""
  filePath= "./FreeThrow.mp4"
  delay = 10

  video = cv2.VideoCapture(filePath)
  ok, frame = video.read()
  while not (cv2.waitKey(delay) & 0xFF == ord('q')):
    cv2.imshow("Tracking", frame)

def main():
  color_segmentation()

if __name__ == "__main__":
  main()