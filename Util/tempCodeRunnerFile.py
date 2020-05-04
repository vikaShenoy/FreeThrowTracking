    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    circleArea = radius * radius * math.pi
    if circleArea > 600:
        return False
    print(circleArea)