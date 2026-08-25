import cv2

CAM_SOURCE = 0

cap = cv2.VideoCapture(CAM_SOURCE, cv2.CAP_MSMF)

if not cap.isOpened():
    raise RuntimeError("Could not open camera. Try changing CAM_SOURCE from 0 to 1.")

try:
    while True:
        ok, frame = cap.read()

        if not ok:
            break

        frame = cv2.flip(frame, 1)
        cv2.imshow("Hello, Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or cv2.getWindowProperty(
            "Hello, Camera", cv2.WND_PROP_VISIBLE
        ) < 1:
            break
finally:
    cap.release()
    cv2.destroyAllWindows()