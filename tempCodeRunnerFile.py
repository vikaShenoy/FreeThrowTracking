        # if not valid_tracker_box(frame, bbox):
        #     inaccuracy_count += 1
        # else:
        #     inaccuracy_count = 0

        # if inaccuracy_count >= inaccuracy_tolerance:
        #     print(f"Tracking failure detected.")
        #     # show(frame)
        #     # Redetect
        #     num, bbox = detect_ball(cap)
        #     if not bbox:
        #         print("Ball could not be re-detected.")
        #         break
        #     cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num + num)
        #     ok, frame = cap.read()
        #     tracker = cv2.TrackerCSRT_create()
        #     tracker.init(frame, bbox)
        #     inaccuracy_count = 0
        #     redetection_count += 1