import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "hand_landmarker.task",
)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

CAM_SOURCE = 0  # Laptop / PC webcam
FRAME_W, FRAME_H = 1280, 720


def ensure_model():
    if os.path.exists(MODEL_PATH):
        return

    print("Downloading hand landmark model (one-time, ~8 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded.")


def main():
    ensure_model()

    if isinstance(CAM_SOURCE, int):
        cap = cv2.VideoCapture(CAM_SOURCE, cv2.CAP_MSMF)
    else:
        cap = cv2.VideoCapture(CAM_SOURCE)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check CAM_SOURCE.")

    print("Hand & Finger AR Tracker running. Press 'q' or Esc to quit.")

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            timestamp_ms = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                timestamp_ms += 1

                for hand_landmarks in result.hand_landmarks:
                    for landmark in hand_landmarks:
                        x = min(int(landmark.x * frame.shape[1]), frame.shape[1] - 1)
                        y = min(int(landmark.y * frame.shape[0]), frame.shape[0] - 1)
                        cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

                    for connection in vision.HandLandmarksConnections.HAND_CONNECTIONS:
                        start = connection.start
                        end = connection.end
                        start_point = hand_landmarks[start].x, hand_landmarks[start].y
                        end_point = hand_landmarks[end].x, hand_landmarks[end].y
                        start_xy = int(start_point[0] * frame.shape[1]), int(start_point[1] * frame.shape[0])
                        end_xy = int(end_point[0] * frame.shape[1]), int(end_point[1] * frame.shape[0])
                        cv2.line(frame, start_xy, end_xy, (0, 255, 0), 2)

                cv2.imshow("Hand & Finger AR Tracker", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27) or cv2.getWindowProperty(
                    "Hand & Finger AR Tracker", cv2.WND_PROP_VISIBLE
                ) < 1:
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()