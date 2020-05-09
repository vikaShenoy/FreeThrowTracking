        # if inaccuracy_count >= inaccuracy_tolerance:
        #     print("Tracking failure detected.")
        #     # show(frame)
        #     # Redetect
        #     bbox = detect_ball(cap)
        #     if not bbox:
        #         print("Ball could not be re-detected.")
        #         break
        #     tracker = cv2.TrackerCSRT_create()
        #     tracker.init(frame, bbox)
        #     inaccuracy_count = 0
        #     continue