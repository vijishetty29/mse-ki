import cv2
import mediapipe as mp
import numpy as np

# Load and resize background
background = cv2.imread("4k-beach-nature-view-s0jgy1y9poz5zg54.jpg")
background = cv2.resize(background, (640, 480))

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))

    # Pose detection on the webcam frame
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    # Copy background for drawing
    output = background.copy()

    # Draw skeleton on the background
    if results.pose_landmarks:
        draw.draw_landmarks(
            image=output,
            landmark_list=results.pose_landmarks,
            connections=mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            connection_drawing_spec=draw.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

    # Show final output
    cv2.imshow("Skeleton on Background", output)

    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
